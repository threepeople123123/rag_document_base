-- 数据库首次初始化时自动执行（空数据卷 + 首次启动）
-- 作用：启用 pgvector 和 zhparser，并配置好中文全文检索

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS zhparser;

-- 中文全文检索配置（词性映射：名词/动词/形容词/习语/叹词/连词）
CREATE TEXT SEARCH CONFIGURATION chinese_zhparser (PARSER = zhparser);
ALTER TEXT SEARCH CONFIGURATION chinese_zhparser ADD MAPPING FOR n,v,a,i,e,l WITH simple;

-- 常用分词选项（对整个集群角色生效）
ALTER ROLE postgres SET zhparser.punctuation_ignore = 'true';
ALTER ROLE postgres SET zhparser.multi_short = 'true';
