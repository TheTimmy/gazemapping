#pragma once
 
#include <vector>
#include <thread>
#include <memory>
#include <future>
#include <functional>
#include <type_traits>
#include <cassert>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cassert>
 
class semaphore
{
public:
    semaphore(int count) noexcept
    : m_count(count) { assert(count > -1); }
 
    void post() noexcept
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            ++m_count;
        }
        m_cv.notify_one();
    }
 
    void wait() noexcept
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&]() { return m_count != 0; });
        --m_count;
    }
 
private:
    int m_count;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};
 
class fast_semaphore
{
public:
    fast_semaphore(int count) noexcept
    : m_count(count), m_semaphore(0) {}
 
    void post()
    {
        std::atomic_thread_fence(std::memory_order_release);
        int count = m_count.fetch_add(1, std::memory_order_relaxed);
        if (count < 0)
            m_semaphore.post();
    }
 
    void wait()
    {
        int count = m_count.fetch_sub(1, std::memory_order_relaxed);
        if (count < 1)
            m_semaphore.wait();
        std::atomic_thread_fence(std::memory_order_acquire);
    }
 
private:
    std::atomic<int> m_count;
    semaphore m_semaphore;
};

template<typename T>
class blocking_queue
{
public:
	blocking_queue(unsigned int size)
	: m_size(size), m_pushIndex(0), m_popIndex(0), m_count(0),
	m_data((T*)operator new(size * sizeof(T))),
	m_openSlots(size), m_fullSlots(0) {}
 
	blocking_queue(const blocking_queue&) = delete;
	blocking_queue(blocking_queue&&) = delete;
	blocking_queue& operator = (const blocking_queue&) = delete;
	blocking_queue& operator = (blocking_queue&&) = delete;
 
	~blocking_queue() noexcept
	{
		while (m_count--)
		{
			m_data[m_popIndex].~T();
			m_popIndex = ++m_popIndex % m_size;
		}
		operator delete(m_data);
	}
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_copy_constructible<Q>::value &&
        std::is_nothrow_copy_constructible<Q>::value, void>::type
    push(const T& item) noexcept
    {
        m_openSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            new (m_data + m_pushIndex) T (item);
            m_pushIndex = ++m_pushIndex % m_size;
            ++m_count;
        }
        m_fullSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_copy_constructible<Q>::value &&
        !std::is_nothrow_copy_constructible<Q>::value, void>::type
	push(const T& item)
	{
		m_openSlots.wait();
		{
			std::lock_guard<std::mutex> lock(m_cs);
			try
			{
				new (m_data + m_pushIndex) T (item);
			}
			catch (...)
			{
				m_openSlots.post();
				throw;
			}
			m_pushIndex = ++m_pushIndex % m_size;
            ++m_count;
		}
		m_fullSlots.post();
	}
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_move_constructible<Q>::value &&
        std::is_nothrow_move_constructible<Q>::value, void>::type
    push(T&& item) noexcept
    {
        m_openSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            new (m_data + m_pushIndex) T (std::move(item));
            m_pushIndex = ++m_pushIndex % m_size;
            ++m_count;
        }
        m_fullSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_move_constructible<Q>::value &&
        !std::is_nothrow_move_constructible<Q>::value, void>::type
    push(T&& item)
    {
        m_openSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            try
            {
                new (m_data + m_pushIndex) T (std::move(item));
            }
            catch (...)
            {
                m_openSlots.post();
                throw;
            }
            m_pushIndex = ++m_pushIndex % m_size;
            ++m_count;
        }
        m_fullSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        !std::is_move_assignable<Q>::value &&
        std::is_nothrow_copy_assignable<Q>::value, void>::type
    pop(T& item) noexcept
    {
        m_fullSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            item = m_data[m_popIndex];
            m_data[m_popIndex].~T();
            m_popIndex = ++m_popIndex % m_size;
            --m_count;
        }
        m_openSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        !std::is_move_assignable<Q>::value &&
        !std::is_nothrow_copy_assignable<Q>::value, void>::type
    pop(T& item)
    {
        m_fullSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            try
            {
                item = m_data[m_popIndex];
            }
            catch (...)
            {
                m_fullSlots.post();
                throw;
            }
            m_data[m_popIndex].~T();
            m_popIndex = ++m_popIndex % m_size;
            --m_count;
        }
        m_openSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_move_assignable<Q>::value &&
        std::is_nothrow_move_assignable<Q>::value, void>::type
    pop(T& item) noexcept
    {
        m_fullSlots.wait();
        {
            std::lock_guard<std::mutex> lock(m_cs);
            item = std::move(m_data[m_popIndex]);
            m_data[m_popIndex].~T();
            m_popIndex = ++m_popIndex % m_size;
            --m_count;
        }
        m_openSlots.post();
    }
 
    template<typename Q = T>
    typename std::enable_if<
        std::is_move_assignable<Q>::value &&
        !std::is_nothrow_move_assignable<Q>::value, void>::type
	pop(T& item)
	{
		m_fullSlots.wait();
		{
			std::lock_guard<std::mutex> lock(m_cs);
			try
			{
                item = std::move(m_data[m_popIndex]);
			}
			catch (...)
			{
				m_fullSlots.post();
				throw;
			}
			m_data[m_popIndex].~T();
			m_popIndex = ++m_popIndex % m_size;
            --m_count;
		}
		m_openSlots.post();
	}
 
    T pop() noexcept(std::is_nothrow_invocable_r<void, decltype(&blocking_queue<T>::pop<T>), T&>::value) {
        T item;
        pop(item);
        return item;
    }
 
    bool empty() noexcept
    {
        std::lock_guard<std::mutex> lock{m_cs};
        return m_count == 0;
    }
 
    bool size() noexcept
    {
        std::lock_guard<std::mutex> lock{m_cs};
        return m_count;
    }
 
    unsigned int max_size() const noexcept
    {
        return m_size;
    }
 
private:
	const unsigned int m_size;
	unsigned int m_pushIndex;
	unsigned int m_popIndex;
    unsigned int m_count;
	T* m_data;
 
    fast_semaphore m_openSlots;
	fast_semaphore m_fullSlots;
    std::mutex m_cs;
};

class thread_pool
{
public:
    using Proc = std::function<void(void)>;
 
    template<typename F, typename... Args>
    void enqueue_work(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable<decltype(&blocking_queue<Proc>::push<Proc&&>)>::value)
    {
        m_workQueue.push([=]() { f(args...); });
    }
 
    template<typename F, typename... Args>
    auto enqueue_task(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        m_workQueue.push([task](){ (*task)(); });
        return res;
    }

    const size_t threadCount() const { return m_threads.size(); }
    const bool initialized() const { return m_threads.size() > 0; }
    void initialize(unsigned int queueDepth = std::thread::hardware_concurrency(), size_t threads = std::thread::hardware_concurrency()) {
        for(size_t i = 0; i < threads; ++i)
            m_threads.emplace_back(std::thread([this]() {
                while(true)
                {
                    auto workItem = m_workQueue.pop();
                    if(workItem == nullptr)
                    {
                        m_workQueue.push(nullptr);
                        break;
                    }
                    workItem();
                }
        }));
    }

     static inline thread_pool& construct(int threads) {
        static thread_pool pool = thread_pool(threads, threads);
        return pool;
    }

    static inline thread_pool& instance() {
/*#ifdef DEBUG
        static thread_pool pool = thread_pool(1, 1);
#else
        static thread_pool pool = thread_pool(8, 8);
#endif*/
        static thread_pool pool = thread_pool();
        return pool;
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator = (const thread_pool&) = delete;
 
private:
    thread_pool(
        unsigned int queueDepth = std::thread::hardware_concurrency(),
        size_t threads = std::thread::hardware_concurrency())
    : m_workQueue(queueDepth)
    {
        if (queueDepth != 0 || threads != 0) {
            assert(queueDepth != 0);
            assert(threads != 0);
            initialize(queueDepth, threads);           
        }
    }
 
    ~thread_pool() noexcept
    {
        m_workQueue.push(nullptr);
        for(auto& thread : m_threads)
            thread.join();
    }

    using ThreadPool = std::vector<std::thread>;
    ThreadPool m_threads;
    blocking_queue<Proc> m_workQueue;
};
