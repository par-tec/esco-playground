ld_dir_all('/initdb.d', '*.ttl', 'http://data.europa.eu/esco');
rdf_loader_run();
exec('checkpoint');
-- WAIT_FOR_CHILDREN;
