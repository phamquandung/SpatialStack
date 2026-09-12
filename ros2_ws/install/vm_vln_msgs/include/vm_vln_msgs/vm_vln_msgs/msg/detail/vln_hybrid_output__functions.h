// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#ifndef VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__FUNCTIONS_H_
#define VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "vm_vln_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "vm_vln_msgs/msg/detail/vln_hybrid_output__struct.h"

/// Initialize msg/VlnHybridOutput message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * vm_vln_msgs__msg__VlnHybridOutput
 * )) before or use
 * vm_vln_msgs__msg__VlnHybridOutput__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__init(vm_vln_msgs__msg__VlnHybridOutput * msg);

/// Finalize msg/VlnHybridOutput message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
void
vm_vln_msgs__msg__VlnHybridOutput__fini(vm_vln_msgs__msg__VlnHybridOutput * msg);

/// Create msg/VlnHybridOutput message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * vm_vln_msgs__msg__VlnHybridOutput__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
vm_vln_msgs__msg__VlnHybridOutput *
vm_vln_msgs__msg__VlnHybridOutput__create();

/// Destroy msg/VlnHybridOutput message.
/**
 * It calls
 * vm_vln_msgs__msg__VlnHybridOutput__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
void
vm_vln_msgs__msg__VlnHybridOutput__destroy(vm_vln_msgs__msg__VlnHybridOutput * msg);

/// Check for msg/VlnHybridOutput message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__are_equal(const vm_vln_msgs__msg__VlnHybridOutput * lhs, const vm_vln_msgs__msg__VlnHybridOutput * rhs);

/// Copy a msg/VlnHybridOutput message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__copy(
  const vm_vln_msgs__msg__VlnHybridOutput * input,
  vm_vln_msgs__msg__VlnHybridOutput * output);

/// Initialize array of msg/VlnHybridOutput messages.
/**
 * It allocates the memory for the number of elements and calls
 * vm_vln_msgs__msg__VlnHybridOutput__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__init(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array, size_t size);

/// Finalize array of msg/VlnHybridOutput messages.
/**
 * It calls
 * vm_vln_msgs__msg__VlnHybridOutput__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
void
vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array);

/// Create array of msg/VlnHybridOutput messages.
/**
 * It allocates the memory for the array and calls
 * vm_vln_msgs__msg__VlnHybridOutput__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
vm_vln_msgs__msg__VlnHybridOutput__Sequence *
vm_vln_msgs__msg__VlnHybridOutput__Sequence__create(size_t size);

/// Destroy array of msg/VlnHybridOutput messages.
/**
 * It calls
 * vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
void
vm_vln_msgs__msg__VlnHybridOutput__Sequence__destroy(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array);

/// Check for msg/VlnHybridOutput message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__are_equal(const vm_vln_msgs__msg__VlnHybridOutput__Sequence * lhs, const vm_vln_msgs__msg__VlnHybridOutput__Sequence * rhs);

/// Copy an array of msg/VlnHybridOutput messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vm_vln_msgs
bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__copy(
  const vm_vln_msgs__msg__VlnHybridOutput__Sequence * input,
  vm_vln_msgs__msg__VlnHybridOutput__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__FUNCTIONS_H_
