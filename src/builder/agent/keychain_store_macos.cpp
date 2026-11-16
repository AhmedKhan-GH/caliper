// The macOS Keychain implementation of KeyStore, through the Security
// framework's SecItem functions. A key is a generic password item:
//   kSecAttrService     "caliper.agent-keys"
//   kSecAttrAccount     the provider id ("anthropic")
//   kSecAttrAccessible  kSecAttrAccessibleWhenUnlockedThisDeviceOnly
//   kSecAttrSynchronizable false (never sent to other devices)
// Every Error names the OSStatus and never the key. No automated test calls this
// file's functions, because touching the login keychain can show a system
// prompt; it is compiled on macOS only.
#include <caliper_builder/agent/key_store.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

#include <string>

namespace caliper_builder::agent {

namespace {

// Owns one Core Foundation reference.
template <typename T>
class CfRef {
public:
    explicit CfRef(T ref = nullptr) : ref_(ref) {}
    ~CfRef() {
        if (ref_) CFRelease(ref_);
    }
    CfRef(const CfRef&) = delete;
    CfRef& operator=(const CfRef&) = delete;
    T get() const { return ref_; }
    T* out() { return &ref_; }

private:
    T ref_;
};

CFStringRef make_string(const std::string& s) {
    return CFStringCreateWithBytes(kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
                                   static_cast<CFIndex>(s.size()), kCFStringEncodingUTF8, false);
}

Error status_error(const char* action, OSStatus status) {
    return {"keystore.failed",
            std::string("The Keychain could not ") + action + " the key (OSStatus " + std::to_string(status) + ")."};
}

// The query that names one item: class, service and account.
CFMutableDictionaryRef item_query(const std::string& provider) {
    CFMutableDictionaryRef q = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                                         &kCFTypeDictionaryValueCallBacks);
    CfRef<CFStringRef> service(make_string(kKeychainService));
    CfRef<CFStringRef> account(make_string(provider));
    CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
    CFDictionarySetValue(q, kSecAttrService, service.get());
    CFDictionarySetValue(q, kSecAttrAccount, account.get());
    return q;
}

class KeychainKeyStore final : public KeyStore {
protected:
    Error do_set(const std::string& provider, const std::string& key) override {
        CfRef<CFMutableDictionaryRef> query(item_query(provider));
        CfRef<CFDataRef> data(CFDataCreate(kCFAllocatorDefault, reinterpret_cast<const UInt8*>(key.data()),
                                           static_cast<CFIndex>(key.size())));
        CfRef<CFMutableDictionaryRef> update(CFDictionaryCreateMutable(
            kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        CFDictionarySetValue(update.get(), kSecValueData, data.get());
        OSStatus status = SecItemUpdate(query.get(), update.get());
        if (status == errSecSuccess) return {};
        if (status != errSecItemNotFound) return status_error("update", status);

        CfRef<CFStringRef> label(make_string("Caliper agent key (" + provider + ")"));
        CFDictionarySetValue(query.get(), kSecValueData, data.get());
        CFDictionarySetValue(query.get(), kSecAttrLabel, label.get());
        CFDictionarySetValue(query.get(), kSecAttrAccessible, kSecAttrAccessibleWhenUnlockedThisDeviceOnly);
        CFDictionarySetValue(query.get(), kSecAttrSynchronizable, kCFBooleanFalse);
        status = SecItemAdd(query.get(), nullptr);
        if (status != errSecSuccess) return status_error("store", status);
        return {};
    }

    KeyPresence do_has(const std::string& provider) override {
        CfRef<CFMutableDictionaryRef> query(item_query(provider));
        CFDictionarySetValue(query.get(), kSecReturnAttributes, kCFBooleanTrue);
        CFDictionarySetValue(query.get(), kSecMatchLimit, kSecMatchLimitOne);
        CfRef<CFTypeRef> result;
        const OSStatus status = SecItemCopyMatching(query.get(), result.out());
        if (status == errSecSuccess) return {{}, true};
        if (status == errSecItemNotFound) return {{}, false};
        return {status_error("look up", status), false};
    }

    Error do_remove(const std::string& provider) override {
        CfRef<CFMutableDictionaryRef> query(item_query(provider));
        const OSStatus status = SecItemDelete(query.get());
        if (status == errSecSuccess || status == errSecItemNotFound) return {};
        return status_error("remove", status);
    }

    Error do_read(const std::string& provider, std::string* out) override {
        CfRef<CFMutableDictionaryRef> query(item_query(provider));
        CFDictionarySetValue(query.get(), kSecReturnData, kCFBooleanTrue);
        CFDictionarySetValue(query.get(), kSecMatchLimit, kSecMatchLimitOne);
        CfRef<CFTypeRef> result;
        const OSStatus status = SecItemCopyMatching(query.get(), result.out());
        if (status == errSecItemNotFound) return {"keystore.not_found", "No key is set for " + provider + "."};
        if (status != errSecSuccess) return status_error("read", status);
        if (!result.get() || CFGetTypeID(result.get()) != CFDataGetTypeID())
            return {"keystore.failed", "The Keychain returned something that is not a key."};
        CFDataRef data = static_cast<CFDataRef>(result.get());
        const CFIndex size = CFDataGetLength(data);
        if (size <= 0 || static_cast<std::size_t>(size) > kMaxKeyBytes)
            return {"keystore.bad_key", "The stored key is not well formed; enter it again."};
        out->assign(reinterpret_cast<const char*>(CFDataGetBytePtr(data)), static_cast<std::size_t>(size));
        return {};
    }
};

}  // namespace

std::unique_ptr<KeyStore> make_system_key_store() {
    return std::make_unique<KeychainKeyStore>();
}

}  // namespace caliper_builder::agent
