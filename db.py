from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import text
from os import getenv
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(getenv('POSTGRESQL_CONNECTION'))
Base = declarative_base()


class Exam(Base):
    __tablename__ = 'exam'

    exam_id = Column(String(16), primary_key=True)
    scores = Column(MutableList.as_mutable(ARRAY(Integer, dimensions=2)))


class Variant(Base):
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