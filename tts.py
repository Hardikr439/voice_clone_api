import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import chatterbox
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# # English example
# model = ChatterboxTTS.from_pretrained(device="cuda")
# AUDIO_PROMPT_PATH = "WhatsApp Ptt 2025-12-11 at 01.32.52.wav"
# conds=model.prepare_conditionals(wav_fpath=AUDIO_PROMPT_PATH,exaggeration=0.7)
# print(conds)
# model.conds=conds
# text = "DFS, or Depth-First Search, is a graph-traversal algorithm that explores as far as possible along each path before backtracking. It uses a stack—either the call stack through recursion or an explicit stack—to track nodes. Starting from a chosen node, it goes deep into one branch, backtracks when it hits a dead end, then explores the next branch. DFS is great for detecting cycles, pathfinding in puzzles, and topological sorting."
# wav=model.generate(text)
# ta.save("test-english_striver.wav", wav, model.sr)

# text = "DFS, or Depth-First Search, is a graph-traversal algorithm that explores as far as possible along each path before backtracking. It uses a stack—either the call stack through recursion or an explicit stack—to track nodes. Starting from a chosen node, it goes deep into one branch, backtracks when it hits a dead end, then explores the next branch. DFS is great for detecting cycles, pathfinding in puzzles, and topological sorting."
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-english_striver.wav", wav, model.sr)

# # # Multilingual examples
# multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
# text = "नमस्ते, आज का दिन बहुत अच्छा है। मुझे उम्मीद है कि आपका दिन भी शानदार हो। चलो कुछ नया सीखते हैं और अपने ज्ञान को बढ़ाते हैं। "
# wav = multilingual_model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH,language_id="hi")
# ta.save("test-striver.wav", wav, model.sr)
help(chatterbox.tts)
