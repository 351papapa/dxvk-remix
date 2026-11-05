/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "log.h"
#include "log_strings.h"

#include "../config/global_options.h"
#include "util_filesys.h"
#include "util_process.h"

#include <array>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace dxvk::util;

namespace bridge_util {
  template<int N>
  static inline void getLocalTimeString(char (&timeString)[N]) {
    // [HH:MM:SS.MS]
    static const char* format = "[%02d:%02d:%02d.%03d] ";

#ifdef _WIN32
    SYSTEMTIME lt;
    GetLocalTime(&lt);

    sprintf_s(timeString, format,
              lt.wHour, lt.wMinute, lt.wSecond, lt.wMilliseconds);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm* lt = localtime(&tv.tv_sec);

    sprintf_s(timeString, format,
              lt->tm_hour, lt->tm_min, lt->tm_sec, (tv.tv_usec / 1000) % 1000);
#endif
  }

  Logger* Logger::logger = nullptr;
  Logger::PreInitMessageArr Logger::s_preInitMsgs;
  std::mutex Logger::s_mutex;

  void Logger::init() {
    assert(logger == nullptr && "Logger already init!");
    if (logger == nullptr) {
      logger = new Logger(GlobalOptions::getLogLevel());
      emitPreInitMsgs();
    }
  }

  Logger& Logger::get() {
    return *logger;
  }

  // A log file stops taking anything but errors at this size, and errors stop kErrorReserveBytes later.
  static constexpr uint64_t kMaxLogBytes = 20ull << 20;
  static constexpr uint64_t kErrorReserveBytes = 1ull << 20;

#ifdef REMIX_BRIDGE_CLIENT
  static constexpr DWORD kLogFileFlags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH;
  static constexpr const char* kLineEnd = "\n";
#else
  static constexpr DWORD kLogFileFlags = FILE_ATTRIBUTE_NORMAL;
  static constexpr const char* kLineEnd = "\r\n";
#endif

  // Opens a log file for appending, creating it when it does not exist.
  // Other processes can read, rename and delete the file while it is open, but not write to it.
  static HANDLE openForAppend(const std::filesystem::path& path) {
    return CreateFileW(path.c_str(), FILE_APPEND_DATA | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_DELETE,
                       NULL, OPEN_ALWAYS, kLogFileFlags, NULL);
  }

  // Returns true when a game integration rotates the logs in this folder itself.
  // Its session.lock stands while the game runs, and its archive.pending while the *.prev.log files wait for their zip.
  static bool logsRotatedByGame(const std::filesystem::path& folder) {
    return GetFileAttributesW((folder / L"session.lock").c_str()) != INVALID_FILE_ATTRIBUTES ||
           GetFileAttributesW((folder / L"archive.pending").c_str()) != INVALID_FILE_ATTRIBUTES;
  }

  Logger::Logger(const LogLevel log_level) : m_level(log_level) {
    if (m_level != LogLevel::None) {

#ifdef REMIX_BRIDGE_CLIENT
      std::string logName = "bridge32.log";
#else
      std::string logName = "bridge64.log";
#endif
      auto logPath = RtxFileSys::path(RtxFileSys::Logs);
      logPath /= logName;

      // The previous session's log is kept as <name>.prev.log, so a relaunch does not lose a crashed session.
      // A game integration that rotates the logs itself has moved it already.
      // The move replaces the previous file only when it succeeds, so a failed move keeps both files.
      if (!logsRotatedByGame(logPath.parent_path())) {
        auto prevPath = logPath;
        prevPath.replace_extension(".prev.log");
        MoveFileExW(logPath.wstring().c_str(), prevPath.wstring().c_str(), MOVEFILE_REPLACE_EXISTING);
      }

      // An existing log is never truncated: a log that is still there goes on below a separator line.
      // A log another process still writes to, such as a bridge server that outlived its game, is left to it,
      // and this process writes <name>.<process id>.log instead.
      const DWORD pid = GetCurrentProcessId();
      std::string note;
      HANDLE file = INVALID_HANDLE_VALUE;
      for (uint32_t attempt = 0; attempt < 4 && file == INVALID_HANDLE_VALUE; attempt++) {
        file = openForAppend(logPath);
        if (file == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION) {
          auto pidPath = logPath;
          pidPath.replace_extension("." + std::to_string(pid) + ".log");
          file = openForAppend(pidPath);
          note = format_string("[Log] %s is still open in another process, so process %lu writes its log to %s in the same folder.",
                               logPath.filename().string().c_str(), pid, pidPath.filename().string().c_str());
        }
        if (file == INVALID_HANDLE_VALUE) {
          emitMsg(LogLevel::Error, format_string("Log CreateFile() failed with %d", GetLastError()));
        }
      }
      if (file == INVALID_HANDLE_VALUE) {
        return;
      }
      m_hFile = file;

      LARGE_INTEGER size {};
      if (GetFileSizeEx(file, &size) && size.QuadPart > 0) {
        emitNote(LogLevel::Info, format_string("[Log] ---- Process %lu continues this log here. The lines above are from an earlier process. ----", pid));
      }
      if (!note.empty()) {
        emitNote(LogLevel::Warn, note);
      }
    }
  }

  Logger::~Logger() {
    if (m_hFile != nullptr) {
      CloseHandle(m_hFile);
    }
  }

  void Logger::emitNote(const LogLevel level, const std::string& message) {
    std::stringstream ss = formatMessage(level, message);
    std::string line;
    while (std::getline(ss, line, '\n')) {
      emitLine(level, line);
    }
  }

  void Logger::emitPreInitMsgs() {
    debug("[Pre-Init Message] Emitting...");
    for(size_t logLevel = (size_t)LogLevel::Trace;
               logLevel < (size_t)LogLevel::None; logLevel++) {
      auto& preInitSS = s_preInitMsgs[logLevel];
      emitMsg((LogLevel)logLevel, preInitSS.str());
      preInitSS.clear();
    }
    debug("[Pre-Init Message] Done!");
  }

  void Logger::trace(const std::string& message) {
    emitMsg(LogLevel::Trace, message);
  }

  void Logger::debug(const std::string& message) {
    emitMsg(LogLevel::Debug, message);
  }

  void Logger::info(const std::string& message) {
    emitMsg(LogLevel::Info, message);
  }

  void Logger::warn(const std::string& message) {
    emitMsg(LogLevel::Warn, message);
  }

  void Logger::err(const std::string& message) {
    emitMsg(LogLevel::Error, message);
  }

  void Logger::errLogMessageBoxAndExit(const std::string& message) {
    Logger::err(message);
    MessageBox(nullptr, message.c_str(), logger_strings::RtxRemixRuntimeError, MB_OK | MB_TOPMOST | MB_TASKMODAL);
    std::exit(-1 );
  }

  void Logger::log(const LogLevel level, const std::string& message) {
    emitMsg(level, message);
  }

  void Logger::logLine(const LogLevel level, const char* line) {
    get().emitLine(level, line);
  }

  void Logger::emitMsg(const LogLevel level, const std::string& message) {
    // Filter early — formatMessage builds a std::stringstream whose locale-facet
    // path can fault on a freed _Locimp; skip it for messages we'd drop anyway.
    if (logger && level < logger->m_level) {
      return;
    }
    if(!logger) {
      auto ss = formatMessage(level, message);
      std::scoped_lock lock(s_mutex);
      s_preInitMsgs[(size_t)level] << ss.str();
    } else {
      auto ss = formatMessage(level, message);
      std::scoped_lock lock(s_mutex);
      std::string line;
      while (std::getline(ss, line, '\n')) {
        logger->emitLine(level, line);
      }
    }
  }

  void Logger::emitLine(const LogLevel level, const std::string& line) {
    if (level >= m_level) {
      if (m_capReached) {
        if (level < LogLevel::Error || m_bytesWritten >= kMaxLogBytes + kErrorReserveBytes) {
          return;
        }
      } else if (m_bytesWritten >= kMaxLogBytes) {
        m_capReached = true;
        emitLine(LogLevel::Error, "err:   [Log] This log reached 20 MB. From here on only errors are written.");
        if (level < LogLevel::Error) {
          return;
        }
      }
      const std::string text = line + kLineEnd;
      m_bytesWritten += text.size();
#if defined(REMIX_BRIDGE_CLIENT) && defined(_DEBUG)
      OutputDebugStringA(text.c_str());
#endif
      // Each line goes to the file in one write, so a crash right after it does not lose it.
      if (m_hFile != nullptr) {
        DWORD written = 0;
        WriteFile(m_hFile, text.data(), static_cast<DWORD>(text.size()), &written, NULL);
      }
    }
  }
  
  std::stringstream Logger::formatMessage(const LogLevel level, const std::string& message) {
    std::stringstream unformattedStream(message);
    std::string       line;
    constexpr std::array<const char*, 5> s_prefixes = {
      "trace: ",
      "debug: ",
      "info:  ",
      "warn:  ",
      "err:   ",
    };
    const char* prefix = s_prefixes[static_cast<uint32_t>(level)];
    char timeString[64];
    getLocalTimeString(timeString);

    std::stringstream formattedStream;
    while (std::getline(unformattedStream, line, '\n')) {
      formattedStream << timeString << prefix << line << '\n';
    }
    return std::move(formattedStream);
  }

  void Logger::set_loglevel(const LogLevel level) {
    get().m_level = level;
  }

}
