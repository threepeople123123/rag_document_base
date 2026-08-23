# ============================================================================
# PostgreSQL 17 + pgvector(官方镜像预装, 支持 HNSW) + zhparser(中文分词)
# 构建镜像:  docker build -t pg-rag:17 .
#
# 要点：
#  - 多阶段构建：builder 编译 scws + zhparser，最终阶段只拷贝产物
#  - bookworm 没有 libscws-dev 包，scws 1.2.3 从源码编译（zhparser 上游官方做法）
#  - apt 源换成阿里云镜像，解决国内拉包慢/超时
#  - 想换 PG 18：docker build --build-arg PG_MAJOR=18 -t pg-rag:18 .
# ============================================================================
ARG PG_MAJOR=17

# ---------- 构建阶段：编译 scws + zhparser ----------
FROM postgres:${PG_MAJOR}-bookworm AS builder
ARG PG_MAJOR

# 换阿里云 apt 镜像源（deb.debian.org / security / pgdg）
RUN set -ex \
    && (sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g; s|http://security.debian.org|http://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources \
        || sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g; s|http://security.debian.org|http://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list \
        || true) \
    && (sed -i 's|http://apt.postgresql.org/pub/repos/apt|http://mirrors.aliyun.com/postgresql/repos/apt|g' /etc/apt/sources.list.d/pgdg.list || true) \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        pkg-config \
        binutils \
        automake \
        libtool \
        postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# 编译安装 scws 1.2.3（zhparser 的分词核心库）
# 注意：git 拉下来的源码没有 configure，需要先 autoreconf 生成
RUN set -ex \
    && git clone --branch 1.2.3 --single-branch --depth 1 https://github.com/hightman/scws.git /tmp/scws \
    && cd /tmp/scws \
    && touch README \
    && aclocal && autoconf && autoheader && libtoolize && automake --add-missing \
    && ./configure \
    && make -j$(nproc) install \
    && rm -rf /tmp/scws

# 编译安装 zhparser（make install 会把 .so、extension SQL、词典都装进 PG 目录）
RUN set -ex \
    && git clone --branch master --single-branch --depth 1 https://github.com/amutu/zhparser.git /tmp/zhparser \
    && cd /tmp/zhparser \
    && make install \
    && rm -rf /tmp/zhparser

# ---------- 最终阶段：pgvector 官方镜像（PG + pgvector 预装） ----------
FROM pgvector/pgvector:pg${PG_MAJOR}
ARG PG_MAJOR

# 从 builder 拷贝 zhparser 全部产物
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/zhparser.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/local/lib/libscws.* /usr/local/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/zhparser* /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/tsearch_data/dict.utf8.xdb /usr/share/postgresql/${PG_MAJOR}/tsearch_data/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/tsearch_data/rules.utf8.ini /usr/share/postgresql/${PG_MAJOR}/tsearch_data/

# 注册 /usr/local/lib 动态库（libscws.so.1 运行时要加载）
RUN ldconfig

# 官方 postgres 镜像入口脚本会在 /docker-entrypoint-initdb.d/ 下执行
# 首次初始化（空数据卷）时自动创建扩展和分词配置
COPY init-db.sql /docker-entrypoint-initdb.d/init-db.sql
