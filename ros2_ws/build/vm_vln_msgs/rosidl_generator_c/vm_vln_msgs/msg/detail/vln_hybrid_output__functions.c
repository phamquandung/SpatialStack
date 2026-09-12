// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice
#include "vm_vln_msgs/msg/detail/vln_hybrid_output__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `trajectory`
#include "nav_msgs/msg/detail/path__functions.h"
// Member `action_codes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
vm_vln_msgs__msg__VlnHybridOutput__init(vm_vln_msgs__msg__VlnHybridOutput * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    vm_vln_msgs__msg__VlnHybridOutput__fini(msg);
    return false;
  }
  // step_index
  // type_of_output
  // trajectory
  if (!nav_msgs__msg__Path__init(&msg->trajectory)) {
    vm_vln_msgs__msg__VlnHybridOutput__fini(msg);
    return false;
  }
  // action_codes
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->action_codes, 0)) {
    vm_vln_msgs__msg__VlnHybridOutput__fini(msg);
    return false;
  }
  return true;
}

void
vm_vln_msgs__msg__VlnHybridOutput__fini(vm_vln_msgs__msg__VlnHybridOutput * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // step_index
  // type_of_output
  // trajectory
  nav_msgs__msg__Path__fini(&msg->trajectory);
  // action_codes
  rosidl_runtime_c__int32__Sequence__fini(&msg->action_codes);
}

bool
vm_vln_msgs__msg__VlnHybridOutput__are_equal(const vm_vln_msgs__msg__VlnHybridOutput * lhs, const vm_vln_msgs__msg__VlnHybridOutput * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // step_index
  if (lhs->step_index != rhs->step_index) {
    return false;
  }
  // type_of_output
  if (lhs->type_of_output != rhs->type_of_output) {
    return false;
  }
  // trajectory
  if (!nav_msgs__msg__Path__are_equal(
      &(lhs->trajectory), &(rhs->trajectory)))
  {
    return false;
  }
  // action_codes
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->action_codes), &(rhs->action_codes)))
  {
    return false;
  }
  return true;
}

bool
vm_vln_msgs__msg__VlnHybridOutput__copy(
  const vm_vln_msgs__msg__VlnHybridOutput * input,
  vm_vln_msgs__msg__VlnHybridOutput * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // step_index
  output->step_index = input->step_index;
  // type_of_output
  output->type_of_output = input->type_of_output;
  // trajectory
  if (!nav_msgs__msg__Path__copy(
      &(input->trajectory), &(output->trajectory)))
  {
    return false;
  }
  // action_codes
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->action_codes), &(output->action_codes)))
  {
    return false;
  }
  return true;
}

vm_vln_msgs__msg__VlnHybridOutput *
vm_vln_msgs__msg__VlnHybridOutput__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vm_vln_msgs__msg__VlnHybridOutput * msg = (vm_vln_msgs__msg__VlnHybridOutput *)allocator.allocate(sizeof(vm_vln_msgs__msg__VlnHybridOutput), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(vm_vln_msgs__msg__VlnHybridOutput));
  bool success = vm_vln_msgs__msg__VlnHybridOutput__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
vm_vln_msgs__msg__VlnHybridOutput__destroy(vm_vln_msgs__msg__VlnHybridOutput * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    vm_vln_msgs__msg__VlnHybridOutput__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__init(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vm_vln_msgs__msg__VlnHybridOutput * data = NULL;

  if (size) {
    data = (vm_vln_msgs__msg__VlnHybridOutput *)allocator.zero_allocate(size, sizeof(vm_vln_msgs__msg__VlnHybridOutput), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = vm_vln_msgs__msg__VlnHybridOutput__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        vm_vln_msgs__msg__VlnHybridOutput__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      vm_vln_msgs__msg__VlnHybridOutput__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

vm_vln_msgs__msg__VlnHybridOutput__Sequence *
vm_vln_msgs__msg__VlnHybridOutput__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vm_vln_msgs__msg__VlnHybridOutput__Sequence * array = (vm_vln_msgs__msg__VlnHybridOutput__Sequence *)allocator.allocate(sizeof(vm_vln_msgs__msg__VlnHybridOutput__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = vm_vln_msgs__msg__VlnHybridOutput__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
vm_vln_msgs__msg__VlnHybridOutput__Sequence__destroy(vm_vln_msgs__msg__VlnHybridOutput__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__are_equal(const vm_vln_msgs__msg__VlnHybridOutput__Sequence * lhs, const vm_vln_msgs__msg__VlnHybridOutput__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!vm_vln_msgs__msg__VlnHybridOutput__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
vm_vln_msgs__msg__VlnHybridOutput__Sequence__copy(
  const vm_vln_msgs__msg__VlnHybridOutput__Sequence * input,
  vm_vln_msgs__msg__VlnHybridOutput__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(vm_vln_msgs__msg__VlnHybridOutput);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    vm_vln_msgs__msg__VlnHybridOutput * data =
      (vm_vln_msgs__msg__VlnHybridOutput *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!vm_vln_msgs__msg__VlnHybridOutput__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          vm_vln_msgs__msg__VlnHybridOutput__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!vm_vln_msgs__msg__VlnHybridOutput__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
