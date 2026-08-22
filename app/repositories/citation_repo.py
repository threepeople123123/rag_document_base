from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base
from app.db.models import AnswerCitation


class AnswerCitationRepository(Base):
    def __init__(self, session:AsyncSession):
        self.session = session

    async def bulk_add(self,citations:Sequence[AnswerCitation])->None:
        if not citations:
            return

        self.session.add_all(citations)
        return await self.session.flush()
