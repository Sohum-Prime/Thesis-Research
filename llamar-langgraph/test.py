import os, sys, json
import glob, imageio

os.chdir("C:/Users/skothavade/OneDrive - Olin College of Engineering/Documents/GitHub/PocketFlow-Template-Python/LLaMAR/LLaMAR/SAR")
sys.path.insert(0, os.getcwd())
print(os.getcwd())

from misc import Arg

key_path = os.path.join(os.getcwd(), "openai_key.json")
with open(key_path) as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]

print(api_key)

# -------- Create GIF from frames -----------------------------------------
frame_paths = sorted(glob.glob('C:/Users/skothavade/OneDrive - Olin College of Engineering/Documents/GitHub/PocketFlow-Template-Python/LLaMAR/LLaMAR/SAR/baselines/results/llamar_SAR/render/2_agents/seed_0/scene_1/frame_*.png'))
if frame_paths:
    images = [imageio.imread(fp) for fp in frame_paths]
    gif_path = 'C:/Users/skothavade/OneDrive - Olin College of Engineering/Documents/GitHub/PocketFlow-Template-Python/LLaMAR/LLaMAR/SAR/baselines/results/llamar_SAR/render/2_agents/seed_0/scene_1/run_through.gif'
    imageio.mimsave(gif_path, images, duration=1.2)
    print(f"Animated GIF saved to {gif_path}")
else:
    print("No frame images found to build GIF.")