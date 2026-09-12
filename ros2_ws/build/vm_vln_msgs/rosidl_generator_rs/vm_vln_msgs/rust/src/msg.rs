#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to vm_vln_msgs__msg__VlnHybridOutput

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VlnHybridOutput {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub step_index: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub type_of_output: u8,

    /// When mode == trajectory → use this field
    pub trajectory: nav_msgs::msg::Path,

    /// When mode == discrete_action → use this field
    pub action_codes: Vec<i32>,

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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::VlnHybridOutput::default())
  }
}

impl rosidl_runtime_rs::Message for VlnHybridOutput {
  type RmwMsg = super::msg::rmw::VlnHybridOutput;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        step_index: msg.step_index,
        type_of_output: msg.type_of_output,
        trajectory: nav_msgs::msg::Path::into_rmw_message(std::borrow::Cow::Owned(msg.trajectory)).into_owned(),
        action_codes: msg.action_codes.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      step_index: msg.step_index,
      type_of_output: msg.type_of_output,
        trajectory: nav_msgs::msg::Path::into_rmw_message(std::borrow::Cow::Borrowed(&msg.trajectory)).into_owned(),
        action_codes: msg.action_codes.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      step_index: msg.step_index,
      type_of_output: msg.type_of_output,
      trajectory: nav_msgs::msg::Path::from_rmw_message(msg.trajectory),
      action_codes: msg.action_codes
          .into_iter()
          .collect(),
    }
  }
}


