"""create document hash on Chunks Table

Revision ID: e548028b4ff2
Revises: 08778a3efc97
Create Date: 2026-02-05 02:21:10.575862
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "e548028b4ff2"
down_revision = "08778a3efc97"
branch_labels = None
depends_on = None


"""
Add content_hash to chunks and deduplicate by (document_id, content_hash).

- Adds column as nullable
- Backfills existing rows using sha256(normalized_content)
- Deletes duplicates within the same document_id
- Sets NOT NULL
- Adds UNIQUE(document_id, content_hash)
"""


def upgrade() -> None:
    # 1) Add column nullable first (safe for existing rows)
    op.add_column(
        "chunks",
        sa.Column("content_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )

    # 2) Backfill existing rows (Postgres)
    # We use: trim(regexp_replace(content, '\s+', ' ', 'g'))
    # which matches Python: re.sub(r"\s+", " ", content).strip()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    op.execute(
        r"""
        UPDATE chunks
        SET content_hash = encode(
            digest(
                convert_to(
                    trim(regexp_replace(content, '\s+', ' ', 'g')),
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        )
        WHERE content_hash IS NULL;
        """
    )

    # 3) Remove duplicates (keep the earliest by chunk_index)
    # Only necessary if duplicates exist; safe to run regardless.
    op.execute(
        """
        DELETE FROM chunks a
        USING (
            SELECT ctid,
                   row_number() OVER (
                       PARTITION BY document_id, content_hash
                       ORDER BY chunk_index
                   ) AS rn
            FROM chunks
            WHERE content_hash IS NOT NULL
        ) b
        WHERE a.ctid = b.ctid
          AND b.rn > 1;
        """
    )

    # 4) Set NOT NULL + add unique constraint
    op.alter_column("chunks", "content_hash", nullable=False)

    op.create_unique_constraint(
        "uq_chunks_document_content_hash",
        "chunks",
        ["document_id", "content_hash"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_chunks_document_content_hash", "chunks", type_="unique")
    op.drop_column("chunks", "content_hash")
