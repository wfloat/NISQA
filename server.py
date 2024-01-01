from pydantic import BaseModel, Field
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from nisqa.NISQA_model import nisqaModel
from util.io import empty_directory
from dataclasses import dataclass

WEIGHTS_PATH = "weights"
OUTPUT_DIR = "output"
PORT = 5239

app = FastAPI()


pretrained_model_options = Literal["nisqa", "nisqa_mos_only", "nisqa_tts"]
class PredictFileArgs(BaseModel):
    pretrained_model: pretrained_model_options = Field(..., description="Model for quality prediction. Accepted values are: 'nisqa' for Overall Quality, Noisiness, Coloration, Discontinuity, Loudness. 'nisqa_mos_only' for Overall Quality only (for finetuning/transfer learning. 'nisqa_tts' only Naturalness prediction.")
    deg: str = Field(default="path/to/file.wav", description="path to speech .wav file.")

class PredictFileResponse(BaseModel):
    deg: str
    mos_pred: float
    noi_pred: Optional[float] = None
    dis_pred: Optional[float] = None
    col_pred: Optional[float] = None
    loud_pred: Optional[float] = None
    model: str

# python run_predict.py --mode predict_file --pretrained_model weights/nisqa.tar --deg /path/to/wav/file.wav --output_dir /path/to/dir/with/results
@app.post("/predict_file")
async def predict_file(args: PredictFileArgs) -> PredictFileResponse:
    deg = f"{WEIGHTS_PATH}/{args.pretrained_model}.tar"
    nisqaArgs = {
        "mode": "predict_file",
        "pretrained_model": deg,
        "deg": args.deg,
        "output_dir": OUTPUT_DIR,
        "bs": 1,
        "num_workers": 0,
        "ms_channel": None,
    }
    nisqaArgs['tr_bs_val'] = nisqaArgs['bs']
    nisqaArgs['tr_num_workers'] = nisqaArgs['num_workers']

    nisqa = nisqaModel(nisqaArgs)
    response_df = nisqa.predict()
    response: PredictFileResponse = response_df.to_dict(orient='records')[0]

    empty_directory(OUTPUT_DIR)
    return response

# @app.post("/predict_dir")
# async def predict_file(args: PredictDirArgs):

# @app.post("/predict_csv")
# async def predict_file(args: PredictCsvArgs):

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)