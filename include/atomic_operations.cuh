/**
 * @file Wrappers for CUDA atomic operations
 *
 * Copyright (c) 2018, Eyal Rozenberg, CWI Amsterdam
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of CWI Amsterdam nor the names of its contributors may
 *    be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
#pragma once
#ifndef CUDA_ON_DEVICE_ATOMICS_CUH_
#define CUDA_ON_DEVICE_ATOMICS_CUH_

#define __fd__    __forceinline__ __device__

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <functional>
#include <type_traits>

namespace atomic {

namespace detail {

/**
 * Use CUDA intrinsics where possible and relevant to reinterpret the bits
 * of values of different types
 *
 * @param x[in]  the value to reinterpret. No references please!
 * @return the reinterpreted value
 */
template <typename ToInterpret, typename Interpreted>
__fd__  Interpreted reinterpret(
	typename std::enable_if<
		!std::is_same<
			typename std::decay<ToInterpret>::type, // I actually just don't want references here
			typename std::decay<Interpreted>::type>::value && // I actually just don't want references here
		sizeof(ToInterpret) == sizeof(Interpreted), ToInterpret>::type x)
{
	return x;
}

template<> __fd__ double reinterpret<long long int, double>(long long int x) { return __longlong_as_double(x); }
template<> __fd__ long long int reinterpret<double, long long int>(double x) { return __double_as_longlong(x); }

template<> __fd__ double reinterpret<unsigned long long int, double>(unsigned long long int x) { return __longlong_as_double(x); }
template<> __fd__ unsigned long long int reinterpret<double, unsigned long long int>(double x) { return __double_as_longlong(x); }

template<> __fd__ float reinterpret<int, float>(int x) { return __int_as_float(x); }
template<> __fd__ int reinterpret<float, int>(float x) { return __float_as_int(x); }

// The default (which should be 32-or-less-bit types
template <typename T, typename = void>
struct add_impl {
	__fd__ T operator()(T*  __restrict__ address, const T& val) const
	{
		return atomicAdd(address, val);
	}
};

template <typename T>
struct add_impl<T,
	typename std::enable_if<
		!std::is_same<T, unsigned long long int>::value &&
		sizeof(T) == sizeof(unsigned long long int)
	>::type> {
	using surrogate_t = unsigned long long int;

	__fd__ T operator()(T*  __restrict__ address, const T& val) const
	{
		auto address_ = reinterpret_cast<surrogate_t*>(address);

		// TODO: Use apply_atomically

		surrogate_t previous_ = *address_;
		surrogate_t expected_previous_;
		do {
			expected_previous_ = previous_;
			T updated_value = reinterpret<surrogate_t, T>(previous_) + val;
			previous_ = atomicCAS(address_, expected_previous_,
				reinterpret<T, surrogate_t>(updated_value));
		} while (expected_previous_ != previous_);
		T rv = reinterpret<surrogate_t, T>(previous_);
		return rv;
	}
};

} // namespace detail

template <typename T>
__fd__ T add(T* __restrict__ address, const T& val)
{
	return detail::add_impl<T>()(address, val);
}


} // namespace atomic

#undef __fd__ 

#endif /* CUDA_ON_DEVICE_ATOMICS_CUH_ */


