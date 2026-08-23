# PostgreSQL 17 + pgvector + zhparser（Docker）

一键启动一个自带 **pgvector（支持 HNSW）** 和 **zhparser（中文分词）** 的 PostgreSQL 17。

## 启动

```bash
cd docker/postgres

# 设置强密码并后台启动（首次会编译 zhparser，约需几分钟）
POSTGRES_PASSWORD=你的强密码 docker compose up -d --build

# 查看启动日志（看到 "database system is ready to accept connections" 即成功）
docker compose logs -f pg-rag
```

## 验证

```bash
# 1) zhparser 中文分词
docker compose exec pg-rag psql -U postgres -d rag_fts \
  -c "SELECT to_tsvector('chinese_zhparser', '南京市的天气怎么样');"

# 2) pgvector HNSW 索引（1024 维，与项目 EMBEDDING_DIMENSIONS 一致）
docker compose exec pg-rag psql -U postgres -d rag_fts \
  -c "CREATE TABLE _t(id bigserial PRIMARY KEY, e vector(1024)); \
      CREATE INDEX ON _t USING hnsw (e vector_cosine_ops); DROP TABLE _t;"
```

## 连接参数

| 项 | 值 |
|---|---|
| 主机 | 运行 docker 的服务器 IP |
| 端口 | **5433**（宿主机；若宿主机 5432 没被旧 PG 占用，可在 compose 里改回 5432） |
| 库名 | `rag_fts` |
| 用户 | `postgres` |
| 密码 | 你设置的 `POSTGRES_PASSWORD` |

对应项目的 `.env`：

```ini
DATABASE_URL=postgresql+asyncpg://postgres:你的强密码@服务器IP:5433/rag_fts
```

## 常用命令

```bash
docker compose down        # 停止（数据保留在 pgdata 卷）
docker compose down -v     # 停止并删除数据卷（重置数据库）
docker compose up -d       # 之后再次启动（镜像已构建，秒起）
```

## 说明

- 镜像基于官方 `pgvector/pgvector:pg17`（PostgreSQL 17 + pgvector 预装），
  zhparser 在镜像构建时源码编译安装；想用 PG 18 只需把 Dockerfile 第一行改成 `pgvector/pgvector:pg18`。
- `init-db.sql` 只在**数据卷首次初始化**时执行；若已启动过再改它不会生效，需要 `docker compose down -v` 重置。
- 中文检索配置名为 `chinese_zhparser`，用法示例：
  `SELECT * FROM t WHERE to_tsvector('chinese_zhparser', content) @@ to_tsquery('chinese_zhparser', '南京');`
