

-- Drop table

-- DROP TABLE public.query_sequence;

CREATE TABLE public.query_sequence (
	"token" varchar(100) NULL,
	txid int8 NULL,
	status int4 NULL
);


-- Drop table

-- DROP TABLE public.progressive_attrs;

CREATE TABLE public.progressive_attrs (
	tbl_name varchar(100) NOT NULL,
	attr_name varchar(100) NOT NULL,
	attr_type varchar(15) NULL,
	num_labels int4 NULL,
	upi int4 NULL,
	CONSTRAINT progressive_attrs_pkey PRIMARY KEY (tbl_name, attr_name)
);

-- Drop table

-- DROP TABLE public.progressive_decision;

CREATE TABLE public.progressive_decision (
	tbl_name varchar(100) NULL,
	attr_name varchar(100) NULL,
	state int4[] NULL,
	uncertainty_ranges float8[] NULL,
	next_func int4[] NULL,
	delta_uncertainty float8[] NULL
);
CREATE INDEX d_idx ON public.progressive_decision USING btree (tbl_name, attr_name);

-- Drop table

-- DROP TABLE public.progressive_function_classes;

CREATE TABLE public.progressive_function_classes (
	id int4 NULL,
	"name" varchar(200) NULL
);

-- Drop table

-- DROP TABLE public.progressive_functions;

CREATE TABLE public.progressive_functions (
	fcid int4 NULL,
	fid serial NOT NULL,
	tbl_name varchar(200) NULL,
	attr_name varchar(200) NULL,
	model_tbl_name varchar(200) NULL,
	bitmap_index int4 NULL,
	"cost" float8 NULL,
	quality float8 NULL,
	parameters text NULL,
	CONSTRAINT progressive_functions_pkey PRIMARY KEY (fid)
);

-- Drop table

-- DROP TABLE public.tweets;

CREATE TABLE public.tweets (
	id int4 NULL,
	tweet_object text NULL,
	feature float8[] NULL,
	topic int4 NULL,
	sentiment int4 NULL,
	"timestamp" timestamp NULL,
	"location" text NULL,
	username text NULL
);
CREATE INDEX tweets_id ON public.tweets USING btree (id);

-- Drop table

-- DROP TABLE public.tweets_full;

CREATE TABLE public.tweets_full (
	id int4 NOT NULL,
	tweet_object text NULL,
	topic int4 NULL,
	sentiment int4 NULL,
	"timestamp" timestamp NULL,
	"location" varchar(50) NULL,
	username varchar(100) NULL,
	CONSTRAINT tweets_full_pkey PRIMARY KEY (id)
);

-- Drop table

-- DROP TABLE public.tweets_state;

CREATE TABLE public.tweets_state (
	id int4 NOT NULL,
	"attribute" varchar(50) NOT NULL,
	state_bitmap int4[] NULL,
	"output" float8[] NULL,
	output_upi jsonb[] NULL,
	CONSTRAINT tweets_state_pkey PRIMARY KEY (id, attribute)
);


INSERT INTO public.progressive_attrs (tbl_name,attr_name,attr_type,num_labels,upi) VALUES 
('wifi','id','precise',NULL,NULL)
,('wifi','timestamp','precise',NULL,NULL)
,('wifi','wifi_ap','precise',NULL,NULL)
,('wifi','location','imprecise',304,NULL)
,('wifi','mac','precise',NULL,NULL)
,('images','id','precise',NULL,0)
,('images','image_blob','precise',NULL,0)
,('images','feature','precise',NULL,0)
,('images','gender','imprecise',2,0)
,('images','age','precise',NULL,0)
;
INSERT INTO public.progressive_attrs (tbl_name,attr_name,attr_type,num_labels,upi) VALUES 
('images','timestamp','precise',NULL,0)
,('images','location','precise',NULL,0)
,('images','cameraid','precise',NULL,0)
,('tweets','id','precise',NULL,0)
,('tweets','tweet_object','precise',NULL,0)
,('tweets','feature','precise',NULL,0)
,('tweets','timestamp','precise',NULL,0)
,('tweets','location','precise',NULL,0)
,('tweets','username','precise',NULL,0)
,('imagenet','id','precise',NULL,NULL)
;
INSERT INTO public.progressive_attrs (tbl_name,attr_name,attr_type,num_labels,upi) VALUES 
('imagenet','feature','precise',NULL,NULL)
,('imagenet','object','imprecise',100,NULL)
,('tweets','sentiment','imprecise',3,0)
,('synthetic','a1','imprecise',4,2)
,('tweets','topic','imprecise',40,NULL)
,('images','expression','imprecise',5,0)
;INSERT INTO public.progressive_decision (tbl_name,attr_name,state,uncertainty_ranges,next_func,delta_uncertainty) VALUES 
('wifi','location','{0,0}','{1}','{9}','{0.25}')
,('wifi','location','{0,1}','{1}','{9}','{0.2}')
,('wifi','location','{1,0}','{1}','{10}','{0.3}')
,('images','gender','{1,0,0,0}','{1}','{14}','{0.025}')
,('images','gender','{1,1,0,0}','{1}','{16}','{0.025}')
,('images','gender','{1,0,1,0}','{1}','{14}','{0.025}')
,('images','gender','{1,0,0,1}','{1}','{15}','{0.025}')
,('images','gender','{1,1,1,0}','{1}','{16}','{0.025}')
,('images','gender','{1,1,0,1}','{1}','{15}','{0.025}')
,('images','gender','{1,0,1,1}','{1}','{14}','{0.025}')
;
INSERT INTO public.progressive_decision (tbl_name,attr_name,state,uncertainty_ranges,next_func,delta_uncertainty) VALUES 
('imagenet','object','{1,0,0,0}','{1}','{20}','{0.025}')
,('imagenet','object','{1,1,0,0}','{1}','{20}','{0.025}')
,('imagenet','object','{1,0,1,0}','{1}','{20}','{0.025}')
,('imagenet','object','{1,0,0,1}','{1}','{19}','{0.025}')
,('imagenet','object','{1,1,1,0}','{1}','{20}','{0.025}')
,('imagenet','object','{1,1,0,1}','{1}','{19}','{0.025}')
,('imagenet','object','{1,0,1,1}','{1}','{18}','{0.025}')
,('sentiment','object','{1,0,0,0}','{1}','{24}','{0.025}')
,('sentiment','object','{1,1,0,0}','{1}','{24}','{0.025}')
,('sentiment','object','{1,0,1,0}','{1}','{24}','{0.025}')
;
INSERT INTO public.progressive_decision (tbl_name,attr_name,state,uncertainty_ranges,next_func,delta_uncertainty) VALUES 
('sentiment','object','{1,0,0,1}','{1}','{23}','{0.025}')
,('sentiment','object','{1,1,1,0}','{1}','{24}','{0.025}')
,('sentiment','object','{1,1,0,1}','{1}','{23}','{0.025}')
,('sentiment','object','{1,0,1,1}','{1}','{22}','{0.025}')
,('tweets','sentiment','{1,1,0,0}','{1}','{24}','{0.025}')
,('tweets','sentiment','{1,0,1,0}','{1}','{24}','{0.025}')
,('tweets','sentiment','{1,1,1,0}','{1}','{24}','{0.025}')
,('tweets','sentiment','{1,1,0,1}','{1}','{23}','{0.025}')
,('tweets','sentiment','{1,0,1,1}','{1}','{22}','{0.025}')
,('tweets','sentiment','{1,0,0,0}','{1}','{24}','{0.025}')
;
INSERT INTO public.progressive_decision (tbl_name,attr_name,state,uncertainty_ranges,next_func,delta_uncertainty) VALUES 
('tweets','topic','{1,0,0,0}','{1}','{28}','{0.025}')
,('tweets','sentiment','{1,0,0,1}','{1}','{23}','{0.025}')
,('tweets','topic','{1,1,0,0}','{1}','{29}','{0.025}')
,('tweets','topic','{1,0,1,0}','{1}','{28}','{0.025}')
,('tweets','topic','{1,0,0,1}','{1}','{28}','{0.025}')
,('tweets','topic','{1,1,1,0}','{1}','{30}','{0.025}')
,('tweets','topic','{1,1,0,1}','{1}','{29}','{0.025}')
,('tweets','topic','{1,0,1,1}','{1}','{28}','{0.025}')
,('images','expression','{1,0,0,1}','{1}','{33}','{0.022}')
,('images','expression','{1,1,0,0}','{1}','{34}','{0.022}')
;
INSERT INTO public.progressive_decision (tbl_name,attr_name,state,uncertainty_ranges,next_func,delta_uncertainty) VALUES 
('images','expression','{1,0,1,0}','{1}','{32}','{0.022}')
,('images','expression','{1,1,1,0}','{1}','{34}','{0.022}')
,('images','expression','{1,1,0,1}','{1}','{33}','{0.022}')
,('images','expression','{1,0,0,0}','{1}','{34}','{0.022}')
,('images','expression','{1,0,1,1}','{1}','{32}','{0.022}')
;INSERT INTO public.progressive_function_classes (id,"name") VALUES 
(1,'madlib.mlp_predict')
,(2,'madlib.svm_predict')
,(3,'madlib.tree_predict')
,(4,'madlib.forest_predict')
,(5,'madlib.lda_predict')
,(6,'madlib.madlib_keras_predict')
,(7,'madlib.logregr_predict')
,(8,'location_room')
,(9,'location_region')
,(11,'gender_gnb')
;
INSERT INTO public.progressive_function_classes (id,"name") VALUES 
(12,'gender_gt')
,(13,'gender_et')
,(14,'gender_rf')
,(15,'imagenet_gnb')
,(16,'imagenet_knn')
,(17,'imagenet_mlp')
,(18,'tweet_dt')
,(19,'tweet_gnb')
,(21,'tweet_mlp')
,(22,'synthetic')
;
INSERT INTO public.progressive_function_classes (id,"name") VALUES 
(10,'imagenet_lda')
,(20,'tweet_lda')
,(23,'topic_dt')
,(24,'topic_gnb')
,(25,'topic_svm')
,(26,'topic_mlp')
,(27,'expression_dt')
,(28,'expression_gnb')
,(29,'expression_mlp')
,(30,'expression_sgd')
;
INSERT INTO public.progressive_function_classes (id,"name") VALUES 
(31,'tweet_sentiment_all')
,(32,'gender_all')
,(33,'imagenet_all')
,(34,'expression_all')
;INSERT INTO public.progressive_functions (fcid,fid, tbl_name,attr_name,model_tbl_name,bitmap_index,"cost",quality,parameters) VALUES
(8,9, 'wifi','location','level1',1,0.8,0.7,'localhost:4567')
,(8,10,'wifi','location','level2',2,0.9,0.9,'localhost:4567')
,(32,53,'images','gender','',2,0.2,0.9,'')
,(34,54,'images','expression','',2,0.2,0.9,'')
,(18,21,'tweets','sentiment','NA',1,0.02,0.6,NULL)
,(19,22,'tweets','sentiment','NA',2,0.04,0.6,NULL)
,(20,23,'tweets','sentiment','NA',3,0.05,0.6,NULL)
,(21,24,'tweets','sentiment','NA',4,0.09,0.6,NULL)
,(23,27,'tweets','topic','NA',1,0.02,0.6,NULL)
,(24,28,'tweets','topic','NA',2,0.04,0.6,NULL)
;
INSERT INTO public.progressive_functions (fcid, fid,tbl_name,attr_name,model_tbl_name,bitmap_index,"cost",quality,parameters) VALUES
(25,29,'tweets','topic','NA',3,0.05,0.6,NULL)
,(26,30,'tweets','topic','NA',4,0.09,0.6,NULL)
,(22,40,'synthetic','a1','NA',1,0.01,0.65,NULL)
,(22,41,'synthetic','a1','NA',2,0.0105,0.7,NULL)
,(22,42,'synthetic','a1','NA',3,0.011,0.78,NULL)
,(22,43,'synthetic','a1','NA',4,0.0115,0.95,NULL)
,(33,51,'imagenet','object','',2,0.2,0.9,'')
;
