import torch
import time
import logging

# Configure logging to write to a text file
logging.basicConfig(filename='gpu_test_log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')


try:

   for i in range(torch.cuda.device_count()):
      print(torch.cuda.get_device_properties(i).name)
      logging.info(f"Device {i}: {torch.cuda.get_device_properties(i).name}")

except Exception as e:
   logging.error(f"Error in getting device properties: {e}")
   print("Error in getting device properties")
   exit(1)