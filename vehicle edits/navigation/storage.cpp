constexpr float kPi = 3.14159265;
constexpr int kLoopFrequency = 20;  // Control loop frequency in Hz
constexpr float kTimeStep = 1.0 / kLoopFrequency;


  // If odometry has not been initialized, we can't do anything.

  static float ct = 0.0;
    
  float L = .5;  // Distance between front and rear axles, adjust based on your vehicle
  float a = 3.0;  // Adjust this value to change the size of the figure 8
  float omega = 1.0 * kPi;  // Adjust this value to change the speed of the figure 8

  // Calculate the desired steering angle
  float delta = atan2(2 * a * cos(omega * ct), L);

  // Calculate the desired velocity
  float velocity = 2.0;  // Constant forward velocity

  // Publish velocity and steering angle
  drive_msg_.velocity = velocity;
  drive_msg_.curvature = delta;

  // The control iteration goes here. 
  // Feel free to make helper functions to structure the control appropriately.
  
  // The latest observed point cloud is accessible via "point_cloud_"

  // Eventually, you will have to set the control values to issue drive commands:

  // IKD CODE:

  // float curvature = 0.8;
  // float velocity = 2.0;
  // float angular_velocity = velocity * curvature;
  // float input_velocity = velocity;
  // std::vector<float> input = {input_velocity, angular_velocity};
  // at::Tensor tensor = torch::from_blob(input.data(), {1, 2}, torch::kFloat32);
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(tensor);
  // std::cout << tensor << std::endl;
  // // run the model
  // at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << output << std::endl;
  // float out = *output[0].data_ptr<float>();
  // // std::cout << out << std::endl;
  // float corr_curvature;
  // if (velocity != 0) {
  // 	corr_curvature = out / velocity;
  // } else {
  // 	corr_curvature = 0;
  // }
  // drive_msg_.velocity = velocity;
  // drive_msg_.curvature = corr_curvature;
  


  // WITHOUT IKD: 

  // float curvature = 0.8;
  // float velocity = 2.0;
  // int counter = 0;
  // int half_fig8_duration = 0; 

  // if half_fig8_duration == 0 {
  //   float circle_radius 
  // }
  // drive_msg_.velocity = velocity;
  // drive_msg_.curvature = curvature;