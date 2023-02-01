from fastapi import FastAPI, UploadFile, File, Form
from uvicorn import run
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from typing import List
from db import *
import io
from ai import EgeModel
import asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="site/static"), name="static")
templates = Jinja2Templates(directory="site/templates")

exam = Exam()
variant = Variant()
solution = Solution()
ai = EgeModel()


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/submit')
def submit_page(request: Request):
    return templates.TemplateResponse('submit.html', {'request': request})


@app.post('/submit')
async def submit_list(request: Request, files: List[UploadFile] = File(...)):
    filenames = []
    for file in files:
        filename = f'./photos/{file.filename}'
        filenames.append(filename)
        file_object = io.BytesIO(await file.read())
        with open(filename, 'wb') as f:
            f.write(file_object.getvalue())
    task = asyncio.create_task(ai.check_blanks(filenames))
    return templates.TemplateResponse('submit.html', {'request': request, 'message': f'{len(files)} файл(ов) успешно сохранены и начинают проверяться'})


@app.get('/manage')
def manage_page(request: Request):
    return templates.TemplateResponse('manage.html', {'request': request})


@app.get('/manage/exams')
def manage_exams(request: Request):
    exams = exam.get_all()
    return templates.TemplateResponse('manage-exams.html', {'request': request, 'exams': exams})


@app.get('/manage/exams/add')
def add_exam_page(request: Request):
    return templates.TemplateResponse('add-exam.html', {'request': request})


@app.get('/manage/variants')
def manage_variants(request: Request):
    variants = variant.get_all()
    return templates.TemplateResponse('manage-variants.html', {'request': request, 'variants': variants})


@app.get('/manage/variants/add')
def add_variant_page(request: Request):
    exam_ids = exam.get_ids()
    return templates.TemplateResponse('add-variant.html', {'request': request, 'exams_ids': exam_ids})


@app.get('/exam-info')
def get_exam_info(request: Request, exam_id: str):
    return exam.get_scores_data(exam_id)


@app.get('/check')
def check_page(request: Request):
    return templates.TemplateResponse('check.html', {'request': request})


@app.post('/check')
def get_result(request: Request, number: str = Form(...)):
    return templates.TemplateResponse('check.html', {'request': request, 'number': number})


@app.post('/add-exam')
def add_exam(request: Request, body: Exam.ApiModel):
    return exam.add(body)


@app.post('/delete-exam')
def delete_exam(request: Request, exam_id: str):
    return exam.delete(exam_id)


@app.post('/add-variant')
def add_variant(request: Request, body: Variant.ApiModel):
    return variant.add(body)


@app.post('/delete-variant')
def delete_variant(request: Request, variant_id: int):
    return variant.delete(variant_id)


if __name__ == '__main__':
    run('server:app', port=5000, reload=True)