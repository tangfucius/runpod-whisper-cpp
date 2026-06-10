from handler import handler
import runpod

runpod.serverless.start({"handler": handler})
