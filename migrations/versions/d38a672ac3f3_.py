"""empty message

Revision ID: d38a672ac3f3
Revises: 61cc7c1ea154
Create Date: 2020-07-15 18:53:31.918030

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd38a672ac3f3'
down_revision = '61cc7c1ea154'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_uploads',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('file', sa.String(length=100), nullable=True),
    sa.Column('diesease', sa.Enum('COVID19', 'Pneumonia', name='dieseasetype'), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('file')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user_uploads')
    # ### end Alembic commands ###
