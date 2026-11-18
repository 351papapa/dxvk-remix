#pragma once

#include "rtx_window_topology.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace dxvk::window_topology_cache {

  using Fingerprint = std::array<uint8_t, 32>;
  using Roots = std::vector<uint32_t>;
  using SharedRoots = std::shared_ptr<const Roots>;

  inline constexpr size_t maxItems = 4096;
  inline constexpr uint32_t maxTriangles = 131072;
  inline constexpr size_t maxFingerprintBytes = 8u * 1024 * 1024;
  inline constexpr size_t maxRootsBytes = 48u * 1024 * 1024;
  inline constexpr size_t maxFileBytes = maxRootsBytes + maxItems * (sizeof(Fingerprint) + sizeof(uint32_t)) + 64;
  inline constexpr std::array<uint8_t, 8> fileVersion { 'V', 'C', 'W', 'I', 'N', '0', '0', '1' };

  inline std::optional<Fingerprint> digest(const void* bytes, size_t size) {
    if (!bytes || size > UINT32_MAX) return std::nullopt;
    Fingerprint result {};
    const auto status = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
      const_cast<PUCHAR>(static_cast<const UCHAR*>(bytes)), static_cast<ULONG>(size), result.data(), static_cast<ULONG>(result.size()));
    if (status < 0) return std::nullopt;
    return result;
  }

  inline bool canRead(const WindowByteSpan& span, uint32_t count, uint32_t stride, uint32_t bytes) {
    return count && stride >= bytes && !span.empty() && uint64_t(count - 1) * stride + bytes <= span.size();
  }

  struct CanonicalInput {
    Fingerprint key;
    WindowTopologyInput layout;
    std::vector<uint8_t> bytes;

    WindowTopologyInput view() const {
      auto in = layout;
      const auto posSize = size_t(in.vertexCount) * 12;
      const auto uvSize = size_t(in.vertexCount) * 8;
      const auto idxSize = size_t(in.indexCount) * in.indexWidth;
      const auto* data = bytes.data() + sizeof(uint32_t) * 5;
      in.positions = {data, posSize};
      in.texcoords = {data + posSize, uvSize};
      in.indices = idxSize ? WindowByteSpan(data + posSize + uvSize, idxSize) : WindowByteSpan {};
      return in;
    }
  };

  // Hash only fields read by buildWindowIslandRoots. Use the same snapshot on a
  // miss so the published fingerprint identifies the bytes actually processed.
  inline std::optional<CanonicalInput> snapshot(const WindowTopologyInput& in) {
    if (!in.triangleCount || in.triangleCount > maxTriangles ||
        (in.topology != WindowPrimitiveTopology::TriangleList && in.topology != WindowPrimitiveTopology::TriangleStrip &&
         in.topology != WindowPrimitiveTopology::TriangleFan) ||
        !canRead(in.positions, in.vertexCount, in.positionStride, 12) ||
        !canRead(in.texcoords, in.vertexCount, in.texcoordStride, 8)) return std::nullopt;
    if (in.indexWidth && ((in.indexWidth != 2 && in.indexWidth != 4) ||
        !canRead(in.indices, in.indexCount, in.indexStride, in.indexWidth))) return std::nullopt;

    const uint32_t layout[] = { in.vertexCount, in.indexCount, in.triangleCount, in.indexWidth, uint32_t(in.topology) };
    const uint64_t bytes = sizeof(layout) + uint64_t(in.vertexCount) * 20 + uint64_t(in.indexCount) * in.indexWidth;
    if (bytes > maxFingerprintBytes) return std::nullopt;
    CanonicalInput result {};
    result.layout = in;
    result.layout.positionStride = 12;
    result.layout.texcoordStride = 8;
    result.layout.indexStride = in.indexWidth;
    auto& canonical = result.bytes;
    canonical.resize(static_cast<size_t>(bytes));
    std::memcpy(canonical.data(), layout, sizeof(layout));
    auto* sink = canonical.data() + sizeof(layout);
    const auto copy = [&](const WindowByteSpan& span, uint32_t count, uint32_t stride, uint32_t size) {
      if (stride == size) {
        const size_t packed = size_t(count) * size;
        std::memcpy(sink, span.data(), packed);
        sink += packed;
      } else {
        for (uint32_t i = 0; i < count; ++i) {
          std::memcpy(sink, span.data() + size_t(i) * stride, size);
          sink += size;
        }
      }
    };
    copy(in.positions, in.vertexCount, in.positionStride, 12);
    copy(in.texcoords, in.vertexCount, in.texcoordStride, 8);
    if (in.indexWidth) copy(in.indices, in.indexCount, in.indexStride, in.indexWidth);
    if (const auto hash = digest(canonical.data(), canonical.size())) {
      result.key = *hash;
      return result;
    }
    return std::nullopt;
  }

  struct KeyHash {
    size_t operator()(const Fingerprint& value) const {
      size_t out = 0;
      std::memcpy(&out, value.data(), sizeof(out));
      return out;
    }
  };

  struct Record {
    Fingerprint key;
    SharedRoots roots;
  };

  inline bool validRoots(const Roots& roots) {
    if (roots.empty() || roots.size() > maxTriangles) return false;
    for (size_t i = 0; i < roots.size(); ++i) {
      const auto root = roots[i];
      if (root > i || roots[root] != root) return false;
    }
    return true;
  }

  inline std::vector<uint8_t> encode(const std::vector<Record>& records) {
    if (records.size() > maxItems) return {};
    size_t payload = sizeof(fileVersion) + sizeof(uint32_t);
    for (const auto& record : records) {
      if (!record.roots || !validRoots(*record.roots)) return {};
      payload += sizeof(Fingerprint) + sizeof(uint32_t) + record.roots->size() * sizeof(uint32_t);
      if (payload > maxFileBytes - sizeof(Fingerprint)) return {};
    }
    std::vector<uint8_t> bytes(payload + sizeof(Fingerprint));
    auto* cursor = bytes.data();
    auto append = [&](const void* part, size_t size) {
      std::memcpy(cursor, part, size);
      cursor += size;
    };
    const auto count = uint32_t(records.size());
    append(fileVersion.data(), fileVersion.size());
    append(&count, sizeof(count));
    for (const auto& record : records) {
      const auto roots = uint32_t(record.roots->size());
      append(record.key.data(), record.key.size());
      append(&roots, sizeof(roots));
      append(record.roots->data(), roots * sizeof(uint32_t));
    }
    const auto hash = digest(bytes.data(), payload);
    if (!hash) return {};
    append(hash->data(), hash->size());
    return bytes;
  }

  inline std::vector<Record> decode(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < sizeof(fileVersion) + sizeof(uint32_t) + sizeof(Fingerprint) || bytes.size() > maxFileBytes ||
        std::memcmp(bytes.data(), fileVersion.data(), fileVersion.size()) != 0) return {};
    const size_t payload = bytes.size() - sizeof(Fingerprint);
    const auto checksum = digest(bytes.data(), payload);
    if (!checksum || std::memcmp(checksum->data(), bytes.data() + payload, checksum->size()) != 0) return {};
    size_t cursor = sizeof(fileVersion);
    uint32_t entries = 0;
    std::memcpy(&entries, bytes.data() + cursor, sizeof(entries));
    cursor += sizeof(entries);
    if (entries > maxItems) return {};
    std::vector<Record> result;
    std::unordered_map<Fingerprint, bool, KeyHash> keys;
    size_t rootsBytes = 0;
    result.reserve(entries);
    for (uint32_t i = 0; i < entries; ++i) {
      Fingerprint key {};
      uint32_t count = 0;
      if (payload - cursor < key.size() + sizeof(count)) return {};
      std::memcpy(key.data(), bytes.data() + cursor, key.size());
      cursor += key.size();
      std::memcpy(&count, bytes.data() + cursor, sizeof(count));
      cursor += sizeof(count);
      const size_t size = size_t(count) * sizeof(uint32_t);
      if (!count || count > maxTriangles || size > payload - cursor || size > maxRootsBytes - rootsBytes ||
          !keys.emplace(key, true).second) return {};
      auto roots = std::make_shared<Roots>(count);
      std::memcpy(roots->data(), bytes.data() + cursor, size);
      if (!validRoots(*roots)) return {};
      cursor += size;
      rootsBytes += size;
      result.push_back({key, std::move(roots)});
    }
    return cursor == payload ? std::move(result) : std::vector<Record>();
  }

  struct Hit {
    SharedRoots roots;
    bool persisted = false;
  };

  class Cache {
    using Sequence = std::list<Fingerprint>;
    struct Entry {
      SharedRoots roots;
      Sequence::iterator age;
      bool persisted = false;
    };

  public:
    explicit Cache(std::filesystem::path path) : m_path(std::move(path)) {
      if (m_path.empty()) return;
      try {
        m_worker = std::thread([this] { service(); });
      } catch (const std::system_error&) {
        // Memory-only operation is sufficient when a worker cannot start.
      }
    }

    ~Cache() {
      if (!m_worker.joinable()) return;
      {
        std::lock_guard lock(m_signalLock);
        m_stop = true;
      }
      m_signal.notify_one();
      m_worker.join();
    }

    Hit find(const Fingerprint& key) {
      std::unique_lock lock(m_dataLock, std::try_to_lock);
      if (!lock.owns_lock()) return {};
      const auto found = m_entries.find(key);
      if (found == m_entries.end()) return {};
      m_age.splice(m_age.end(), m_age, found->second.age);
      return {found->second.roots, found->second.persisted};
    }

    void remember(const Fingerprint& key, const Roots& roots) {
      if (!validRoots(roots)) return;
      auto owned = std::make_shared<Roots>(roots);
      std::unique_lock lock(m_dataLock, std::try_to_lock);
      if (!lock.owns_lock()) return;
      if (const auto found = m_entries.find(key); found != m_entries.end()) {
        if (*found->second.roots == roots) return;
        m_rootsBytes -= found->second.roots->size() * sizeof(uint32_t);
        m_age.erase(found->second.age);
        m_entries.erase(found);
      }
      insert(key, std::move(owned), false);
      ++m_generation;
    }

  private:
    void insert(const Fingerprint& key, SharedRoots roots, bool persisted) {
      const size_t bytes = roots->size() * sizeof(uint32_t);
      while (!m_age.empty() && (m_entries.size() >= maxItems || bytes > maxRootsBytes - m_rootsBytes)) {
        const auto old = m_entries.find(m_age.front());
        m_rootsBytes -= old->second.roots->size() * sizeof(uint32_t);
        m_entries.erase(old);
        m_age.pop_front();
      }
      m_age.push_back(key);
      try {
        m_entries.emplace(key, Entry {std::move(roots), std::prev(m_age.end()), persisted});
        m_rootsBytes += bytes;
      } catch (...) {
        m_age.pop_back();
        throw;
      }
    }

    void load() {
      std::ifstream file(m_path, std::ios::binary | std::ios::ate);
      if (!file) return;
      const auto length = file.tellg();
      if (length <= 0 || uint64_t(length) > maxFileBytes) return;
      std::vector<uint8_t> bytes(static_cast<size_t>(length));
      file.seekg(0);
      if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) return;
      auto records = decode(bytes);
      for (auto& record : records) {
        // Use short locks so a render lookup can miss safely without blocking.
        std::lock_guard lock(m_dataLock);
        if (m_entries.find(record.key) == m_entries.end() && m_entries.size() < maxItems &&
            record.roots->size() * sizeof(uint32_t) <= maxRootsBytes - m_rootsBytes) {
          insert(record.key, std::move(record.roots), true);
        }
      }
    }

    bool persist(uint64_t& saved) {
      std::vector<Record> records;
      uint64_t generation = 0;
      {
        std::lock_guard lock(m_dataLock);
        if (m_generation == saved) return true;
        generation = m_generation;
        records.reserve(m_age.size());
        for (const auto& key : m_age) records.push_back({key, m_entries.at(key).roots});
      }
      const auto bytes = encode(records);
      if (bytes.empty()) return false;
      std::error_code error;
      if (!m_path.parent_path().empty()) std::filesystem::create_directories(m_path.parent_path(), error);
      if (error) return false;
      auto temporary = m_path;
      temporary += L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
      {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (file) file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        file.close();
        if (!file) {
          std::filesystem::remove(temporary, error);
          return false;
        }
      }
      if (!MoveFileExW(temporary.c_str(), m_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::filesystem::remove(temporary, error);
        return false;
      }
      saved = generation;
      return true;
    }

    void service() noexcept {
      try { load(); } catch (...) { }
      uint64_t saved = 0;
      bool stop = false;
      do {
        {
          std::unique_lock lock(m_signalLock);
          m_signal.wait_for(lock, std::chrono::seconds(10), [this] { return m_stop; });
          stop = m_stop;
        }
        try { persist(saved); } catch (...) { }
      } while (!stop);
    }

    const std::filesystem::path m_path;
    std::mutex m_dataLock;
    Sequence m_age;
    std::unordered_map<Fingerprint, Entry, KeyHash> m_entries;
    size_t m_rootsBytes = 0;
    uint64_t m_generation = 0;
    std::mutex m_signalLock;
    std::condition_variable m_signal;
    bool m_stop = false;
    std::thread m_worker;
  };

} // namespace dxvk::window_topology_cache
