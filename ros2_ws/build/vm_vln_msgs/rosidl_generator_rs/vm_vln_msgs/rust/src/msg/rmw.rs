#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "vm_vln_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__vm_vln_msgs__msg__VlnHybridOutput() -> *const std::ffi::c_void;
}

#[link(name = "vm_vln_msgs__rosidl_generator_c")]
extern "C" {
    fn vm_vln_msgs__msg__VlnHybridOutput__init(msg: *mut VlnHybridOutput) -> bool;
    fn vm_vln_msgs__msg__VlnHybridOutput__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<VlnHybridOutput>, size: usize) -> bool;
    fn vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<VlnHybridOutput>);
    fn vm_vln_msgs__msg__VlnHybridOutput__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<VlnHybridOutput>, out_seq: *mut rosidl_runtime_rs::Sequence<VlnHybridOutput>) -> bool;
}

// Corresponds to vm_vln_msgs__msg__VlnHybridOutput
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VlnHybridOutput {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub step_index: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub type_of_output: u8,

    /// When mode == trajectory → use this field
    pub trajectory: nav_msgs::msg::rmw::Path,

    /// When mode == discrete_action → use this field
    pub action_codes: rosidl_runtime_rs::Sequence<i32>,

}

impl VlnHybridOutput {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TRAJECTORY: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const DISCRETE_ACTION: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const STOP: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MOVE_FORWARD: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TURN_LEFT: u8 = 2;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TURN_RIGHT: u8 = 3;

}


impl Default for VlnHybridOutput {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !vm_vln_msgs__msg__VlnHybridOutput__init(&mut msg as *mut _) {
        panic!("Call to vm_vln_msgs__msg__VlnHybridOutput__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for VlnHybridOutput {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vm_vln_msgs__msg__VlnHybridOutput__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vm_vln_msgs__msg__VlnHybridOutput__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { vm_vln_msgs__msg__VlnHybridOutput__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for VlnHybridOutput {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for VlnHybridOutput where Self: Sized {
  const TYPE_NAME: &'static str = "vm_vln_msgs/msg/VlnHybridOutput";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__vm_vln_msgs__msg__VlnHybridOutput() }
  }
}


