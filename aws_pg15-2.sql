 SELECT  table_name FROM information_schema.tables WHERE table_schema = 'public' 
 
 select * from public.user
 
 UPDATE public.user SET org_id = 1 WHERE id = 1;
 
 select * from public.user_data
 select * from public.session_data

 select * from public.organization
 select * from public.tables
 select * from public.table_data
 
 insert into public.organization (id, name, dirpath, is_dropbox) values (1,'Woodland','./OrderForms', True)
 UPDATE public.organization SET dirpath = './OrderForms24' WHERE id = 1;
 UPDATE public.organization SET is_dropbox = True WHERE id = 1;
 UPDATE public.organization SET is_init = True WHERE id = 1;
 UPDATE public.organization SET init_status = 'Inventory last refreshed at Mar 18 06:11 PM' WHERE id = 1;
 UPDATE public.organization SET data = Null WHERE id = 1;

 select * from session_data 
 select * from threads 
 select * from public.user 
 select * from public.tables 
 

 select * from public.organization
 select length(init_details) from public.organization

--- if we want to clean sheet the data 
--- TRUNCATE TABLE public.table_data;
--- TRUNCATE TABLE public.tables;

