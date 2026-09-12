// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "vm_vln_msgs/msg/detail/vln_hybrid_output__rosidl_typesupport_introspection_c.h"
#include "vm_vln_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "vm_vln_msgs/msg/detail/vln_hybrid_output__functions.h"
#include "vm_vln_msgs/msg/detail/vln_hybrid_output__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `trajectory`
#include "nav_msgs/msg/path.h"
// Member `trajectory`
#include "nav_msgs/msg/detail/path__rosidl_typesupport_introspection_c.h"
// Member `action_codes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  vm_vln_msgs__msg__VlnHybridOutput__init(message_memory);
}

void vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_fini_function(void * message_memory)
{
  vm_vln_msgs__msg__VlnHybridOutput__fini(message_memory);
}

size_t vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__size_function__VlnHybridOutput__action_codes(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_const_function__VlnHybridOutput__action_codes(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_function__VlnHybridOutput__action_codes(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__fetch_function__VlnHybridOutput__action_codes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_const_function__VlnHybridOutput__action_codes(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__assign_function__VlnHybridOutput__action_codes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_function__VlnHybridOutput__action_codes(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__resize_function__VlnHybridOutput__action_codes(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vm_vln_msgs__msg__VlnHybridOutput, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "step_index",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vm_vln_msgs__msg__VlnHybridOutput, step_index),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "type_of_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vm_vln_msgs__msg__VlnHybridOutput, type_of_output),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "trajectory",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vm_vln_msgs__msg__VlnHybridOutput, trajectory),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "action_codes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vm_vln_msgs__msg__VlnHybridOutput, action_codes),  // bytes offset in struct
    NULL,  // default value
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__size_function__VlnHybridOutput__action_codes,  // size() function pointer
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_const_function__VlnHybridOutput__action_codes,  // get_const(index) function pointer
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__get_function__VlnHybridOutput__action_codes,  // get(index) function pointer
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__fetch_function__VlnHybridOutput__action_codes,  // fetch(index, &value) function pointer
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__assign_function__VlnHybridOutput__action_codes,  // assign(index, value) function pointer
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__resize_function__VlnHybridOutput__action_codes  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_members = {
  "vm_vln_msgs__msg",  // message namespace
  "VlnHybridOutput",  // message name
  5,  // number of fields
  sizeof(vm_vln_msgs__msg__VlnHybridOutput),
  vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_member_array,  // message members
  vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_init_function,  // function to initialize message memory (memory has to be allocated)
  vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_type_support_handle = {
  0,
  &vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_vm_vln_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vm_vln_msgs, msg, VlnHybridOutput)() {
  vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, nav_msgs, msg, Path)();
  if (!vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_type_support_handle.typesupport_identifier) {
    vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &vm_vln_msgs__msg__VlnHybridOutput__rosidl_typesupport_introspection_c__VlnHybridOutput_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
