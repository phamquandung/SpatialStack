#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <string>
#include <iostream>
#include <thread>

class SetInstructionNode : public rclcpp::Node
{
public:
    SetInstructionNode() : Node("spatialstack_set_instruction_node")
    {
        this->declare_parameter<std::string>("set_instruction_topic", "/vln/input/set_instruction");
        this->declare_parameter<std::string>("task_success_topic", "/vln/output/task_success");
        std::string set_instruction_topic = this->get_parameter("set_instruction_topic").as_string();
        std::string task_success_topic = this->get_parameter("task_success_topic").as_string();

        instruction_pub_ = this->create_publisher<std_msgs::msg::String>(
            set_instruction_topic, 10);

        success_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            task_success_topic, 10,
            std::bind(&SetInstructionNode::taskSuccessCallback, this, std::placeholders::_1));

        input_thread_ = std::thread(&SetInstructionNode::inputLoop, this);
    }

    ~SetInstructionNode()
    {
        running_ = false;
        if (input_thread_.joinable())
            input_thread_.join();
    }

private:
    void inputLoop()
    {
        while (rclcpp::ok() && running_)
        {
            if (!inferring_)
            {
                std::string instruction;
                std::cout << "\nEnter instruction: ";
                std::getline(std::cin, instruction);

                if (instruction.empty())
                    continue;

                auto msg = std_msgs::msg::String();
                msg.data = instruction;
                instruction_pub_->publish(msg);
                RCLCPP_INFO(this->get_logger(), "Published instruction: '%s'", instruction.c_str());
                inferring_ = true;
            }
        }
    }

    void taskSuccessCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data)
        {
            RCLCPP_INFO(this->get_logger(), "Task completed!");
            inferring_ = false;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Thinking and Executing...");
        }
    }

    bool running_ = true;
    std::thread input_thread_;
    bool inferring_ = false;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr instruction_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr success_sub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SetInstructionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
