# import Preprocessor
import time

def preprocessor_benchmark(cfg_path = None):
  time_start = time.time()
  time.sleep(1.234)
  ### calls the actual Preprocessor method
  # Preprocessor.preprocessor_start(cfg_path)
  time_end = time.time()
  print("PLACEHOLDER benchmark time (remember to change commented lines): " + str(time_end - time_start))
  return time_end - time_start
