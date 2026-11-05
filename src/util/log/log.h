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
#pragma once

#include <cstdint>
#include <array>
#include <fstream>
#include <string>
#include <unordered_map>

#include "../thread.h"

namespace dxvk {
  
  enum class LogLevel : std::uint32_t {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    None  = 5,
  };

  /**
   * \brief Logger
   * 
   * Logger for one DLL. Creates a text file and
   * writes all log messages to that file.
   */
  class Logger {
    
  public:

    // NV-DXVK start: pass log level as param
    Logger(const std::string& fileName, const LogLevel logLevel = getMinLogLevel());
    // NV-DXVK end
    ~Logger() = default;
    
    // NV-DXVK start: special init pathway for remix logs
    static void initRtxLog();
    // NV-DXVK end
    
    static void trace(const std::string& message);
    static void debug(const std::string& message);
    static void info (const std::string& message);
    static void warn (const std::string& message);
    static void err  (const std::string& message);
    static void log  (LogLevel level, const std::string& message);
    // An info line the repeat limit never holds back, for lines that are already rate limited by their caller.
    // Like errors, status lines are still written after the log reaches its size cap.
    static void status(const std::string& message);
    
    static LogLevel logLevel() {
      return s_instance.m_minLevel;
    }

    // Warnings and errors since startup, including lines the repeat limit held back.
    static uint64_t warningCount();
    static uint64_t errorCount();
    // Lines the repeat limit held back since startup.
    static uint64_t heldBackCount();

    // Off (the default): a line kind ("[Tag] word") is written freely for its first lines, then at most once
    // every few minutes with a count of the lines held back. On: every line is written.
    static void setWriteAllRepeats(bool enable);

    // Returns the full path the log file really has on disk.
    // A virtual file system, such as a mod manager's, can redirect the file away from the game folder.
    // Returns the path the log was opened at when the real path cannot be read, and an empty string without a log file.
    static const std::wstring& realFilePath() {
      return s_instance.m_realFilePath;
    }

    // Returns true when a virtual file system, such as a mod manager's, is loaded in this process.
    static bool virtualFileSystemActive();

  private:

    static Logger s_instance;

    LogLevel m_minLevel;
    // NV-DXVK start: Don't double print every line
    bool m_doublePrintToStdErr;
    // NV-DXVK end

    dxvk::mutex   m_mutex;
    // The log file's handle, or null without a log file.
    // The handle stays open until the process ends, so lines written during static destruction still reach the file.
    void*         m_file = nullptr;
    std::wstring  m_realFilePath;

    struct RepeatState {
      uint32_t written = 0;
      uint32_t heldBack = 0;
      uint64_t lastWriteMs = 0;
    };
    std::unordered_map<std::string, RepeatState> m_repeats;
    uint64_t m_bytesWritten = 0;
    bool m_capReached = false;

    void emitMsg(LogLevel level, const std::string& message, bool limitRepeats = true);
    bool admitRepeat(LogLevel level, const std::string& message, std::string& note);
    void writeLines(LogLevel level, const std::string& message);
    
    static LogLevel getMinLogLevel();
    static std::string getFilePath(const std::string& fileName);

    Logger& operator=(Logger&& other);

  };
  
}
