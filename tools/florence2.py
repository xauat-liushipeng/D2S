import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

image = Image.open("0001.jpg").convert("RGB")

def run_example(task_prompt, text_input=None):
	if text_input is None:
		prompt = task_prompt
	else:
		prompt = task_prompt + text_input
	inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
	generated_ids = model.generate(
	  input_ids=inputs["input_ids"],
	  pixel_values=inputs["pixel_values"],
	  max_new_tokens=1024,
	  num_beams=3
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

	parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

	print(parsed_answer)

if __name__ == "__main__":
	prompt = "<MORE_DETAILED_CAPTION>"
	run_example(prompt)

	"""
	000005.jpg
	
	{'<MORE_DETAILED_CAPTION>': 
	'The image shows a room with a wooden archway in the center. 
	The archway is made of dark wood and has a clock on top. 
	On the right side of the archway, there is a wooden dresser with a mirror above it. 
	On top of the dresser, there are several cardboard boxes stacked on top of each other. 
	In the center of the room, there appears to be a dining area with a table and chairs. 
	The walls are painted in a light blue color and there are floral wallpaper on the windows. 
	The floor is covered with a blue and white patterned rug. 
	There are also a few other items scattered around the room.'}

	{'<DETAILED_CAPTION>': 
	'The image shows a living room filled with furniture, including a desk, chair, and cabinetry. 
	There are cardboard boxes, shirts, and other objects scattered around the room, as well as a clock on the wall. 
	The room also has windows with curtains, a chandelier, and a carpet on the floor.'}

	{'<CAPTION>': 
	'A living room filled with lots of boxes and furniture.'}

	"""