from fastapi import FastAPI, UploadFile, File
from uvicorn import run
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="site/static"), name="static")
templates = Jinja2Templates(directory="site/templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/submit')
def submit_page(request: Request):
    return templates.TemplateResponse('submit.html', {'request': request})


@app.post('/submit')
def submit_list(request: Request, files: List[UploadFile] = File(...)):
    print(files)
    for file in files:
        print(file.filename)
    return templates.TemplateResponse('submit.html', {'request': request, 'message': f'{len(files)} файл(ов) успешно сохранены'})


if __name__ == '__main__':
    run('server:app', port=5000, reload=True)