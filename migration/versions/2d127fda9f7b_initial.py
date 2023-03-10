"""'initial'

Revision ID: 2d127fda9f7b
Revises: 
Create Date: 2023-01-10 15:32:58.366755

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2d127fda9f7b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('exam',
    sa.Column('exam_id', sa.String(length=16), nullable=False),
    sa.Column('scores', postgresql.ARRAY(sa.Integer(), dimensions=2), nullable=True),
    sa.PrimaryKeyConstraint('exam_id')
    )
    op.create_table('variant',
    sa.Column('variant_id', sa.Integer(), nullable=False),
    sa.Column('answers', postgresql.ARRAY(sa.String(length=16)), nullable=True),
    sa.Column('exam_id', sa.String(length=16), nullable=False),
    sa.ForeignKeyConstraint(['exam_id'], ['exam.exam_id'], ),
    sa.PrimaryKeyConstraint('variant_id')
    )
    op.create_table('solution',
    sa.Column('solution_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=16), nullable=False),
    sa.Column('surname', sa.String(length=16), nullable=False),
    sa.Column('passport', sa.Integer(), nullable=False),
    sa.Column('answers', postgresql.ARRAY(sa.String(length=16)), nullable=True),
    sa.Column('variant_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['variant_id'], ['variant.variant_id'], ),
    sa.PrimaryKeyConstraint('solution_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('solution')
    op.drop_table('variant')
    op.drop_table('exam')
    # ### end Alembic commands ###
