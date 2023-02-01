from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from typing import *
from os import getenv
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import JSONResponse

load_dotenv()
engine = create_engine(getenv('POSTGRESQL_CONNECTION'))
Base = declarative_base()
Sessions = sessionmaker(bind=engine)


class Exam(Base):

    class ApiModel(BaseModel):
        exam_id: str
        scores: List[List[int]]


    __tablename__ = 'exam'

    exam_id = Column(String(16), primary_key=True)
    scores = Column(MutableList.as_mutable(ARRAY(Integer, dimensions=2)))

    def add(self, body: ApiModel):
        with Sessions() as sess:
            if sess.query(Exam).filter_by(exam_id=body.exam_id).first() is not None:
                return JSONResponse(status_code=401, content={'message': 'Экзамен с таким ID уже есть в системе'})
            exam = Exam(exam_id=body.exam_id, scores=body.scores)
            sess.add(exam)
            sess.commit()
        return JSONResponse({'message': 'Экзамен сохранен'})

    
    def delete(self, exam_id: str):
        with Sessions() as sess:
            exam = sess.query(Exam).filter_by(exam_id=exam_id).first()
            sess.delete(exam)
            sess.commit()
        return JSONResponse({'message': f'Экзамен с ID: {exam_id} удален'})


    def get_scores_data(self, exam_id: str):
        with Sessions() as sess:
            exam = sess.query(Exam).filter_by(exam_id=exam_id).first()
            if exam is not None:
                return exam.scores
            else:
                raise FileNotFoundError


    def get_all(self):
        with Sessions() as sess:
            return sess.query(Exam).all()

class Variant(Base):

    class ApiModel(BaseModel):
        variant_id: str
        answers: List[str]
        exam_id: str

    __tablename__ = 'variant'

    variant_id = Column(Integer, primary_key=True)
    answers = Column(MutableList.as_mutable(ARRAY(String(16))))
    exam_id = Column(String(16), ForeignKey('exam.exam_id'), nullable=False)


class Solution(Base):
    __tablename__ = 'solution'

    solution_id = Column(Integer, primary_key=True)
    name = Column(String(16), nullable=False)
    surname = Column(String(16), nullable=False)
    passport = Column(Integer, nullable=False)
    answers = Column(MutableList.as_mutable(ARRAY(String(16))))
    variant_id = Column(Integer, ForeignKey('variant.variant_id'), nullable=False)