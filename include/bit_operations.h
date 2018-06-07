#pragma once
#ifndef BIT_OPERATIONS_H_
#define BIT_OPERATIONS_H_

#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif

// The following would be better placed in a proper bit vector class having per-element proxies.

template <typename Integer>
__fhd__ constexpr Integer get_bit_range(
	const Integer& value,
	unsigned       start_bit,
	unsigned       num_bits) noexcept
{
	return (value >> start_bit) & ((1 << num_bits) - 1);
}

/**
 * @note @p value is assumed to have an appropriate number of bits
 */
template <typename Integer>
__fhd__ constexpr void set_bit_range(
	Integer&  value,
	unsigned  start_bit,
	unsigned  num_bits,
	Integer   sub_value) noexcept
{
	value |= (sub_value << start_bit);
}

template <unsigned LogBitsPerValue, typename Index>
__fhd__ bit_container_t get_bit_resolution_element(
	bit_container_t  bit_container,
	Index            element_index_within_container)
{
	enum {
		bits_per_value = 1 << LogBitsPerValue,
		elements_per_container = bits_per_bit_container / bits_per_value,
	};
	// TODO: This can be improved with some PTX intrinsics
	return get_bit_range<Index>(bit_container,
		bits_per_value * element_index_within_container, bits_per_value);

}

/**
 * @note assumes the relevant bits of @p bit_container are 0
 */
template <unsigned LogBitsPerValue, typename Index>
__fhd__ void set_bit_resolution_element(
	bit_container_t& bit_container,
	Index            element_index_within_container,
	bit_container_t  value)
{
	enum {
		bits_per_value = 1 << LogBitsPerValue,
		elements_per_container = bits_per_bit_container / bits_per_value,
	};
	// TODO: This might be improved with some PTX intrinsics
	set_bit_range<Index>(bit_container,
		bits_per_value * element_index_within_container, bits_per_value, value);

}

template <unsigned LogBitsPerValue, typename Index>
__fhd__ bit_container_t& bit_container_for_element(
	const bit_container_t* __restrict__ data,
	Index                               element_index)
{
	enum {
		bits_per_value = 1 << LogBitsPerValue,
		elements_per_container = bits_per_bit_container / bits_per_value
	};
	auto index_of_container = element_index / elements_per_container;
	return const_cast<bit_container_t&>(data[index_of_container]);
}


template <unsigned LogBitsPerValue, typename Index>
__fhd__ bit_container_t get_bit_resolution_element(
	const bit_container_t* __restrict__ data,
	Index                               element_index)
{
	enum {
		bits_per_value = 1 << LogBitsPerValue,
		elements_per_container = bits_per_bit_container / bits_per_value
	};
	auto index_within_container = element_index % elements_per_container;
	return get_bit_resolution_element<LogBitsPerValue, Index>(
		bit_container_for_element(data, element_index),
		index_within_container);
}

template <unsigned LogBitsPerValue, typename Index>
__fhd__ void set_bit_resolution_element(
	const bit_container_t* __restrict__ data,
	Index                               element_index,
	bit_container_t                     value)
{
	enum {
		bits_per_value = 1 << LogBitsPerValue,
		elements_per_container = bits_per_bit_container / bits_per_value
	};
	auto index_within_container = element_index % elements_per_container;
	auto& container = const_cast<bit_container_t&>(
		bit_container_for_element<LogBitsPerValue, Index>(data, element_index));
	set_bit_resolution_element<LogBitsPerValue, Index>(container, index_within_container, value);
}

#undef __fd__
#undef __fhd__


#endif /* BIT_OPERATIONS_H_ */
