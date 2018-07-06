#ifndef H_ALLOCATOR
#define H_ALLOCATOR

#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>

class Allocator {
private:
	static size_t roundUpToPageSize(size_t s, size_t pagesize) {
		return ((s + pagesize - 1) / pagesize) * pagesize;
	}

	const size_t m_pagesize;

public:
	Allocator() : m_pagesize(getpagesize()) {
	}

	void* alloc(size_t sz, size_t& alloced_size, bool populate) {
		assert(sz > 0);
		alloced_size = roundUpToPageSize(sz, m_pagesize);
		void* p = mmap(0, alloced_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
		if (!p || p == MAP_FAILED) {
			printf("Allocation failed, sz=%d, alloced_size=%d, err=%s\n", (int)sz, (int)alloced_size, strerror(errno));
			return nullptr;
		}
		return p;
	}

	void* realloc(void* old_p, size_t old_alloc_size, size_t new_size, size_t& new_alloc_size) {
		if (!old_p || old_p == MAP_FAILED) {
			return nullptr;
		}
		assert((size_t)old_p % m_pagesize == 0);
		new_alloc_size = roundUpToPageSize(new_size, m_pagesize);
		assert((size_t)new_alloc_size % m_pagesize == 0);
		// FIXME: might be broken
		void* p = mremap(old_p, old_alloc_size, new_alloc_size, MREMAP_MAYMOVE, old_p);
		assert(p);
		if (!p || p == MAP_FAILED) {
			printf("Remap failed old_p=%p, old_alloc_size=%d, new_alloc_size=%d, err=%s\n", old_p, (int)old_alloc_size, (int)new_alloc_size, strerror(errno));
			assert(false);
			return nullptr;
		}
		return p;
	}

	void free(void* p, size_t alloc_size) {
		if (p && p != MAP_FAILED) {
			munmap(p, alloc_size);
		}
	}
};

/* Super-magic in-place-resizable buffer for the age of big data.
 * Utilizing deep-learning techniques this buffer can predict in a PERFECT way
 * the correct buffer size.
 */
template<typename ItemT>
class Buffer {
public:
	ItemT* m_ptr;
	size_t m_capacity;
	size_t m_alloc_size;
	Allocator m_alloc;

public:
	ItemT val;

	Buffer(size_t init_cap, bool init = false) {
	 	m_ptr = (ItemT*)m_alloc.alloc(init_cap * sizeof(ItemT), m_alloc_size, init);
	 	assert(m_ptr);
	 	m_capacity = m_alloc_size / sizeof(ItemT);
	 	assert(m_capacity >= init_cap);
	}

	//Buffer(const Buffer& b) = delete;
	//Buffer(Buffer&& o) = delete;

	//Buffer& operator=(const Buffer&) & = delete;
	//Buffer& operator=(Buffer&&) & = delete;

	~Buffer() {
		m_alloc.free(m_ptr, m_alloc_size);
	}

	void resize(size_t new_cap) {
		size_t new_alloc_size;

		auto p = static_cast<ItemT*>(m_alloc.realloc(m_ptr, m_alloc_size, new_cap * sizeof(ItemT), new_alloc_size));

		m_capacity = new_alloc_size / sizeof(ItemT);
		m_alloc_size = new_alloc_size;
		m_ptr = p;
	}

	void resizeByFactor(float factor) {
		resize((float)m_capacity * factor);
	}

	void set_zero() {
		memset(m_ptr, 0, m_alloc_size);
	}

	size_t size() const noexcept {
		return m_capacity;
	}

	size_t capacity() const noexcept {
		return size();
	}

	size_t alloc_size() const noexcept {
		return m_alloc_size;
	}

	ItemT* get() const noexcept {
		return (ItemT*)__builtin_assume_aligned(m_ptr, 64);
	}
};

#endif