/*
* Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "../util_env.h"
#include "../util_filesys.h"
// NV-DXVK start: Fix some circular inclusion stuff
#include "../util_string.h"
// NV-DXVK end


// NV-DXVK start: Don't double print every line
namespace{
  bool getDoublePrintToStdErr() {
    const std::string str = dxvk::env::getEnvVar("DXVK_LOG_NO_DOUBLE_PRINT_STDERR");
    return str.empty();
  }

  template<int N>
  static inline void getLocalTimeString(char(&timeString)[N]) {
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
}
// NV-DXVK end

namespace {
  // A log file stops taking anything but errors and status lines at this size, and those stop kErrorReserveBytes later.
  constexpr uint64_t kMaxLogBytes = 20ull << 20;
  constexpr uint64_t kErrorReserveBytes = 1ull << 20;
  // A line kind is written freely this many times (errors: kFreeErrorLines), then at most once per kRepeatIntervalMs.
  constexpr uint32_t kFreeLines = 10;
  constexpr uint32_t kFreeErrorLines = 50;
  constexpr uint64_t kRepeatIntervalMs = 5ull * 60ull * 1000ull;
  // Kinds beyond this many are not tracked and always written.
  constexpr size_t kMaxRepeatKinds = 4096;

  std::atomic<uint64_t> s_warningCount { 0 };
  std::atomic<uint64_t> s_errorCount { 0 };
  std::atomic<uint64_t> s_heldBackCount { 0 };
  std::atomic<bool> s_writeAllRepeats { false };

  uint64_t steadyMs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  // The kind of a tagged line: its leading "[Tag]" (or "[Tag] [Sub]") plus the next word, without digits, a
  // leading 0x value or anything after '=' or ':'. An untagged line has no kind and is never limited.
  std::string repeatKind(const std::string& message) {
    if (message.empty() || message[0] != '[') {
      return std::string();
    }
    std::string kind;
    size_t i = 0;
    // The tag, and a second tag right after it.
    for (int tags = 0; tags < 2 && i < message.size() && message[i] == '['; ++tags) {
      const size_t close = message.find(']', i);
      if (close == std::string::npos || close - i > 48) {
        return std::string();
      }
      kind.append(message, i, close - i + 1);
      i = close + 1;
      while (i < message.size() && message[i] == ' ') {
        ++i;
      }
    }
    if (message.compare(i, 2, "0x") == 0) {
      return kind;
    }
    kind.push_back(' ');
    for (size_t n = 0; i < message.size() && n < 24; ++i, ++n) {
      const char c = message[i];
      if (c == ' ' || c == '=' || c == ':' || c == '(' || c == ',' || c == '\n') {
        break;
      }
      if (c < '0' || c > '9') {
        kind.push_back(c);
      }
    }
    return kind;
  }

  // Returns the path the file system really stores the file at, read from an open handle to it.
  // A virtual file system redirects the open, so the handle names the redirected file.
  // Returns an empty string when the file cannot be opened or its path cannot be read.
  std::wstring finalPathOf(const std::wstring& path) {
    const HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      return std::wstring();
    }
    std::wstring result(MAX_PATH, L'\0');
    DWORD len = GetFinalPathNameByHandleW(file, result.data(), DWORD(result.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (len >= result.size()) {
      result.resize(len);
      len = GetFinalPathNameByHandleW(file, result.data(), DWORD(result.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    }
    CloseHandle(file);
    if (len == 0 || len >= result.size()) {
      return std::wstring();
    }
    result.resize(len);

    // The path comes back as \\?\C:\... or \\?\UNC\server\share\..., which Explorer does not take.
    const std::wstring uncPrefix = L"\\\\?\\UNC\\";
    const std::wstring localPrefix = L"\\\\?\\";
    if (result.compare(0, uncPrefix.size(), uncPrefix) == 0) {
      result = L"\\\\" + result.substr(uncPrefix.size());
    } else if (result.compare(0, localPrefix.size(), localPrefix) == 0) {
      result = result.substr(localPrefix.size());
    }
    return result;
  }

  // Opens a log file for appending, creating it when it does not exist.
  // Other processes can read, rename and delete the file while it is open, but not write to it.
  HANDLE openForAppend(const std::filesystem::path& path) {
    return CreateFileW(path.c_str(), FILE_APPEND_DATA | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_DELETE,
                       nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  }

  // Returns true when a game integration rotates the logs in this folder itself.
  // Its session.lock stands while the game runs, and its archive.pending while the *.prev.log files wait for their zip.
  bool logsRotatedByGame(const std::filesystem::path& folder) {
    return GetFileAttributesW((folder / L"session.lock").c_str()) != INVALID_FILE_ATTRIBUTES ||
           GetFileAttributesW((folder / L"archive.pending").c_str()) != INVALID_FILE_ATTRIBUTES;
  }

  // Opens the session log at `path` and returns the path of the file it opened.
  // The log moves to <name>.prev.log first, unless a game integration rotates the logs itself.
  // An existing log is never truncated: a log that is still there goes on below a separator line.
  // A log another process still writes to, such as a renderer that outlived its game, is left to it,
  // and this process writes <name>.<process id>.log instead.
  // Returns a null `file` when no file can be opened. `note` receives the lines for the top of the log.
  std::filesystem::path openSessionLog(const std::filesystem::path& path, HANDLE& file, std::string& note) {
    if (!logsRotatedByGame(path.parent_path())) {
      std::filesystem::path prevPath = path;
      prevPath.replace_extension(L".prev.log");
      // The move replaces the previous file only when it succeeds, so a failed move keeps both files.
      MoveFileExW(path.c_str(), prevPath.c_str(), MOVEFILE_REPLACE_EXISTING);
    }

    const DWORD pid = GetCurrentProcessId();
    std::filesystem::path openedPath = path;
    HANDLE handle = openForAppend(path);
    if (handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION) {
      openedPath.replace_extension(L"." + std::to_wstring(pid) + L".log");
      handle = openForAppend(openedPath);
      note = dxvk::str::format("[Log] ", dxvk::str::fromws(path.filename().c_str()), " is still open in another process, so process ", pid,
                               " writes its log to ", dxvk::str::fromws(openedPath.filename().c_str()), " in the same folder.");
    }
    if (handle == INVALID_HANDLE_VALUE) {
      file = nullptr;
      return openedPath;
    }

    LARGE_INTEGER size {};
    if (GetFileSizeEx(handle, &size) && size.QuadPart > 0) {
      note = dxvk::str::format("[Log] ---- Process ", pid, " continues this log here. The lines above are from an earlier process. ----",
                               note.empty() ? "" : "\n", note);
    }
    file = handle;
    return openedPath;
  }
}

namespace dxvk {

  Logger::Logger(const std::string& fileName, const LogLevel logLevel)
  : m_minLevel(logLevel)
  // NV-DXVK start: Don't double print every line
  , m_doublePrintToStdErr(getDoublePrintToStdErr())
  // NV-DXVK end
  {
    if (m_minLevel != LogLevel::None) {
      const auto path = getFilePath(fileName);

      if (!path.empty()) {
        // The previous session's log is kept as <name>.prev.log, so a relaunch does not lose a crashed session.
        std::string note;
        HANDLE file = nullptr;
        const std::filesystem::path filePath = openSessionLog(std::filesystem::path(str::tows(path.c_str())), file, note);
        m_file = file;

        if (m_file != nullptr) {
          m_realFilePath = finalPathOf(filePath.wstring());
          if (m_realFilePath.empty()) {
            std::error_code ec;
            m_realFilePath = std::filesystem::absolute(filePath, ec).wstring();
          }
          if (!note.empty()) {
            writeLines(LogLevel::Info, note);
          }
        }
      }
    }
  }

  uint64_t Logger::warningCount() {
    return s_warningCount.load(std::memory_order_relaxed);
  }

  uint64_t Logger::errorCount() {
    return s_errorCount.load(std::memory_order_relaxed);
  }

  uint64_t Logger::heldBackCount() {
    return s_heldBackCount.load(std::memory_order_relaxed);
  }

  void Logger::setWriteAllRepeats(bool enable) {
    s_writeAllRepeats.store(enable, std::memory_order_relaxed);
  }
  
  void Logger::initRtxLog() {
    // Proton sets DXVK_LOG_LEVEL=none for every game, so this log treats "none" as the default level and is always written.
    const LogLevel level = getMinLogLevel();
    s_instance = std::move(Logger("remix-dxvk.log", level == LogLevel::None ? LogLevel::Info : level));

    if (!s_instance.m_realFilePath.empty()) {
      const std::filesystem::path folder = std::filesystem::path(s_instance.m_realFilePath).parent_path();
      status(str::format("[Session] log folder: ", str::fromws(folder.c_str()),
                         virtualFileSystemActive() ? " (a virtual file system is active)" : ""));
    }
  }

  bool Logger::virtualFileSystemActive() {
    return GetModuleHandleW(L"usvfs_x64.dll") != nullptr || GetModuleHandleW(L"usvfs_x86.dll") != nullptr;
  }

  void Logger::trace(const std::string& message) {
    s_instance.emitMsg(LogLevel::Trace, message);
  }
  
  void Logger::debug(const std::string& message) {
    s_instance.emitMsg(LogLevel::Debug, message);
  }

  void Logger::info(const std::string& message) {
    s_instance.emitMsg(LogLevel::Info, message);
  }

  void Logger::warn(const std::string& message) {
    s_instance.emitMsg(LogLevel::Warn, message);
  }

  void Logger::err(const std::string& message) {
    s_instance.emitMsg(LogLevel::Error, message);
  }

  void Logger::log(LogLevel level, const std::string& message) {
    s_instance.emitMsg(level, message);
  }

  void Logger::status(const std::string& message) {
    s_instance.emitMsg(LogLevel::Info, message, false);
  }

  void Logger::emitMsg(LogLevel level, const std::string& message, bool limitRepeats) {
    if (level >= m_minLevel) {
      if (level == LogLevel::Warn) {
        s_warningCount.fetch_add(1, std::memory_order_relaxed);
      } else if (level == LogLevel::Error) {
        s_errorCount.fetch_add(1, std::memory_order_relaxed);
      }

      std::lock_guard<dxvk::mutex> lock(m_mutex);

      std::string note;
      if (limitRepeats && !s_writeAllRepeats.load(std::memory_order_relaxed) && !admitRepeat(level, message, note)) {
        s_heldBackCount.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      // Status lines are rate limited by their callers and carry the session lines a bug report needs at its end.
      const bool writtenPastCap = level >= LogLevel::Error || !limitRepeats;
      if (m_capReached) {
        if (!writtenPastCap || m_bytesWritten >= kMaxLogBytes + kErrorReserveBytes) {
          return;
        }
      } else if (m_bytesWritten >= kMaxLogBytes) {
        m_capReached = true;
        writeLines(LogLevel::Error, str::format("[Log] This log reached ", kMaxLogBytes >> 20,
                                                " MB. From here on only errors and status lines are written."));
        if (!writtenPastCap) {
          return;
        }
      }

      if (note.empty()) {
        writeLines(level, message);
      } else {
        writeLines(level, message + note);
      }
    }
  }

  bool Logger::admitRepeat(LogLevel level, const std::string& message, std::string& note) {
    std::string kind = repeatKind(message);
    if (kind.empty()) {
      return true;
    }
    auto it = m_repeats.find(kind);
    if (it == m_repeats.end()) {
      if (m_repeats.size() >= kMaxRepeatKinds) {
        return true;
      }
      it = m_repeats.emplace(std::move(kind), RepeatState {}).first;
    }
    RepeatState& state = it->second;
    const uint64_t now = steadyMs();
    const uint32_t freeLines = level >= LogLevel::Error ? kFreeErrorLines : kFreeLines;
    if (state.written < freeLines) {
      state.written++;
      state.lastWriteMs = now;
      return true;
    }
    if (now - state.lastWriteMs >= kRepeatIntervalMs) {
      if (state.heldBack > 0) {
        note = str::format(" [", state.heldBack, " more lines of this kind held back since the last one]");
      }
      state.written++;
      state.heldBack = 0;
      state.lastWriteMs = now;
      return true;
    }
    if (state.heldBack == 0 && state.written == freeLines) {
      writeLines(LogLevel::Info, str::format("[Log] '", it->first, "' lines are now written at most every ",
                                             kRepeatIntervalMs / 60000, " minutes (rtx.log.writeAllRepeats writes all)."));
    }
    state.heldBack++;
    return false;
  }

  void Logger::writeLines(LogLevel level, const std::string& message) {
    OutputDebugString((message + '\n').c_str());

    constexpr std::array<const char*, 5> s_prefixes{
      "trace: ",
      "debug: ",
      "info:  ",
      "warn:  ",
      "err:   ",
    };
    const char* prefix = s_prefixes[static_cast<std::uint32_t>(level)];

    std::stringstream stream(message);
    std::string       line;

    char timeString[64];
    getLocalTimeString(timeString);

    while (std::getline(stream, line, '\n')) {
      // NV-DXVK start: Don't double print every line
      if (m_doublePrintToStdErr) {
        std::cerr << timeString << prefix << line << std::endl;
      }
      // NV-DXVK end

      // Each line goes to the file in one write, so a crash right after it does not lose it.
      if (m_file != nullptr) {
        std::string text = timeString;
        text += prefix;
        text += line;
        text += "\r\n";
        DWORD written = 0;
        WriteFile(m_file, text.data(), DWORD(text.size()), &written, nullptr);
        m_bytesWritten += text.size();
      }
    }
  }
  
  
  LogLevel Logger::getMinLogLevel() {
    const std::array<std::pair<const char*, LogLevel>, 6> logLevels{ {
      { "trace", LogLevel::Trace },
      { "debug", LogLevel::Debug },
      { "info",  LogLevel::Info  },
      { "warn",  LogLevel::Warn  },
      { "error", LogLevel::Error },
      { "none",  LogLevel::None  },
    } };
    
    const std::string logLevelStr = env::getEnvVar("DXVK_LOG_LEVEL");
    
    for (const auto& pair : logLevels) {
      if (logLevelStr == pair.first)
        return pair.second;
    }
    
    return LogLevel::Info;
  }
  
  std::string Logger::getFilePath(const std::string& fileName) {
    // NV-DXVK start: Use std::filesystem::path helpers + RtxFileSys
    auto path = util::RtxFileSys::path(util::RtxFileSys::Logs);

    // Note: If no path is specified to store log files in, simply use the current directory by returning
    // the specified log file name directly.
    if (path.empty()) {
      return fileName;
    }

    // Append the specified log file name to the logging directory.
    path /= fileName;

    return path.string();
    // NV-DXVK end
  }
  
  Logger& Logger::operator=(Logger&& other) {
    m_minLevel = other.m_minLevel;
    m_doublePrintToStdErr = other.m_doublePrintToStdErr;
    std::swap(m_file, other.m_file);
    std::swap(m_bytesWritten, other.m_bytesWritten);
    std::swap(m_realFilePath, other.m_realFilePath);
    return *this;
  }
  
}
