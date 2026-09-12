# generated from rosidl_generator_py/resource/_idl.py.em
# with input from vm_vln_msgs:msg/VlnHybridOutput.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'action_codes'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_VlnHybridOutput(type):
    """Metaclass of message 'VlnHybridOutput'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'TRAJECTORY': 0,
        'DISCRETE_ACTION': 1,
        'STOP': 0,
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3,
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('vm_vln_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'vm_vln_msgs.msg.VlnHybridOutput')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__vln_hybrid_output
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__vln_hybrid_output
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__vln_hybrid_output
            cls._TYPE_SUPPORT = module.type_support_msg__msg__vln_hybrid_output
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__vln_hybrid_output

            from nav_msgs.msg import Path
            if Path.__class__._TYPE_SUPPORT is None:
                Path.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'TRAJECTORY': cls.__constants['TRAJECTORY'],
            'DISCRETE_ACTION': cls.__constants['DISCRETE_ACTION'],
            'STOP': cls.__constants['STOP'],
            'MOVE_FORWARD': cls.__constants['MOVE_FORWARD'],
            'TURN_LEFT': cls.__constants['TURN_LEFT'],
            'TURN_RIGHT': cls.__constants['TURN_RIGHT'],
        }

    @property
    def TRAJECTORY(self):
        """Message constant 'TRAJECTORY'."""
        return Metaclass_VlnHybridOutput.__constants['TRAJECTORY']

    @property
    def DISCRETE_ACTION(self):
        """Message constant 'DISCRETE_ACTION'."""
        return Metaclass_VlnHybridOutput.__constants['DISCRETE_ACTION']

    @property
    def STOP(self):
        """Message constant 'STOP'."""
        return Metaclass_VlnHybridOutput.__constants['STOP']

    @property
    def MOVE_FORWARD(self):
        """Message constant 'MOVE_FORWARD'."""
        return Metaclass_VlnHybridOutput.__constants['MOVE_FORWARD']

    @property
    def TURN_LEFT(self):
        """Message constant 'TURN_LEFT'."""
        return Metaclass_VlnHybridOutput.__constants['TURN_LEFT']

    @property
    def TURN_RIGHT(self):
        """Message constant 'TURN_RIGHT'."""
        return Metaclass_VlnHybridOutput.__constants['TURN_RIGHT']


class VlnHybridOutput(metaclass=Metaclass_VlnHybridOutput):
    """
    Message class 'VlnHybridOutput'.

    Constants:
      TRAJECTORY
      DISCRETE_ACTION
      STOP
      MOVE_FORWARD
      TURN_LEFT
      TURN_RIGHT
    """

    __slots__ = [
        '_header',
        '_step_index',
        '_type_of_output',
        '_trajectory',
        '_action_codes',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'step_index': 'uint32',
        'type_of_output': 'uint8',
        'trajectory': 'nav_msgs/Path',
        'action_codes': 'sequence<int32>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['nav_msgs', 'msg'], 'Path'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int32')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.step_index = kwargs.get('step_index', int())
        self.type_of_output = kwargs.get('type_of_output', int())
        from nav_msgs.msg import Path
        self.trajectory = kwargs.get('trajectory', Path())
        self.action_codes = array.array('i', kwargs.get('action_codes', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.step_index != other.step_index:
            return False
        if self.type_of_output != other.type_of_output:
            return False
        if self.trajectory != other.trajectory:
            return False
        if self.action_codes != other.action_codes:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def step_index(self):
        """Message field 'step_index'."""
        return self._step_index

    @step_index.setter
    def step_index(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'step_index' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'step_index' field must be an unsigned integer in [0, 4294967295]"
        self._step_index = value

    @builtins.property
    def type_of_output(self):
        """Message field 'type_of_output'."""
        return self._type_of_output

    @type_of_output.setter
    def type_of_output(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'type_of_output' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'type_of_output' field must be an unsigned integer in [0, 255]"
        self._type_of_output = value

    @builtins.property
    def trajectory(self):
        """Message field 'trajectory'."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value):
        if __debug__:
            from nav_msgs.msg import Path
            assert \
                isinstance(value, Path), \
                "The 'trajectory' field must be a sub message of type 'Path'"
        self._trajectory = value

    @builtins.property
    def action_codes(self):
        """Message field 'action_codes'."""
        return self._action_codes

    @action_codes.setter
    def action_codes(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'i', \
                "The 'action_codes' array.array() must have the type code of 'i'"
            self._action_codes = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'action_codes' field must be a set or sequence and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._action_codes = array.array('i', value)
