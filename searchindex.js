Search.setIndex({docnames:["index","readme","reference/components","reference/metrics","reference/models","reference/random","reference/reference"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","readme.rst","reference/components.rst","reference/metrics.rst","reference/models.rst","reference/random.rst","reference/reference.rst"],objects:{"components.base_components":{BaseComponent:[2,1,1,""],BaseObservable:[2,1,1,""],Component:[2,1,1,""],FromNdArray:[2,1,1,""],SystemStateModule:[2,1,1,""],register_observables:[2,4,1,""],unregister_observables:[2,4,1,""]},"components.base_components.BaseComponent":{get_component_state:[2,2,1,""],get_timesteps:[2,2,1,""],observe:[2,2,1,""]},"components.base_components.BaseObservable":{get_observable:[2,2,1,""],observe:[2,2,1,""]},"components.base_components.Component":{store_state:[2,2,1,""]},"components.base_components.SystemStateModule":{_system_state:[2,3,1,""],add_state_variable:[2,2,1,""]},"components.items":{Items:[2,1,1,""]},"components.items.Items":{name:[2,3,1,""]},"components.socialgraph":{BinarySocialGraph:[2,1,1,""]},"components.socialgraph.BinarySocialGraph":{add_friends:[2,2,1,""],follow:[2,2,1,""],remove_friends:[2,2,1,""],unfollow:[2,2,1,""]},"components.users":{ActualUserProfiles:[2,1,1,""],DNUsers:[2,1,1,""],PredictedScores:[2,1,1,""],PredictedUserProfiles:[2,1,1,""],Users:[2,1,1,""]},"components.users.DNUsers":{calc_dn_utilities:[2,2,1,""],get_user_feedback:[2,2,1,""],normalize_values:[2,2,1,""],sample_from_error_dist:[2,2,1,""]},"components.users.Users":{actual_user_profiles:[2,3,1,""],actual_user_scores:[2,3,1,""],compute_user_scores:[2,2,1,""],get_actual_user_scores:[2,2,1,""],get_user_feedback:[2,2,1,""],interact_with_items:[2,3,1,""],repeat_interactions:[2,3,1,""],score_fn:[2,3,1,""],score_new_items:[2,2,1,""],set_score_function:[2,2,1,""],store_state:[2,2,1,""],update_profiles:[2,2,1,""],user_vector:[2,3,1,""]},"metrics.measurement":{DiffusionTreeMeasurement:[3,1,1,""],HomogeneityMeasurement:[3,1,1,""],InteractionMeasurement:[3,1,1,""],MSEMeasurement:[3,1,1,""],Measurement:[3,1,1,""],StructuralVirality:[3,1,1,""]},"metrics.measurement.DiffusionTreeMeasurement":{_old_infection_state:[3,3,1,""],diffusion_tree:[3,3,1,""],draw_tree:[3,2,1,""],measure:[3,2,1,""],name:[3,3,1,""]},"metrics.measurement.HomogeneityMeasurement":{_old_histogram:[3,3,1,""],measure:[3,2,1,""],name:[3,3,1,""]},"metrics.measurement.InteractionMeasurement":{measure:[3,2,1,""],name:[3,3,1,""]},"metrics.measurement.MSEMeasurement":{measure:[3,2,1,""],name:[3,3,1,""]},"metrics.measurement.Measurement":{get_measurement:[3,2,1,""],get_timesteps:[3,2,1,""],measure:[3,2,1,""],measurement_history:[3,3,1,""],name:[3,3,1,""],observe:[3,2,1,""]},"metrics.measurement.StructuralVirality":{get_structural_virality:[3,2,1,""]},"models.bass":{BassModel:[4,1,1,""],InfectionState:[4,1,1,""],InfectionThresholds:[4,1,1,""]},"models.bass.BassModel":{draw_diffusion_tree:[4,2,1,""],get_structural_virality:[4,2,1,""],infection_probabilities:[4,2,1,""],initialize_user_scores:[4,2,1,""],run:[4,2,1,""]},"models.content":{ContentFiltering:[4,1,1,""]},"models.content.ContentFiltering":{process_new_items:[4,2,1,""]},"models.popularity":{PopularityRecommender:[4,1,1,""]},"models.popularity.PopularityRecommender":{process_new_items:[4,2,1,""]},"models.recommender":{BaseRecommender:[4,1,1,""]},"models.recommender.BaseRecommender":{add_new_item_indices:[4,2,1,""],choose_interleaved_items:[4,2,1,""],create_and_process_items:[4,2,1,""],generate_recommendations:[4,2,1,""],get_measurements:[4,2,1,""],get_system_state:[4,2,1,""],indices:[4,3,1,""],initialize_user_scores:[4,2,1,""],interleaving_fn:[4,3,1,""],items:[4,3,1,""],items_hat:[4,3,1,""],measure_content:[4,2,1,""],num_items:[4,3,1,""],num_items_per_iter:[4,3,1,""],num_users:[4,3,1,""],predicted_scores:[4,3,1,""],probabilistic_recommendations:[4,3,1,""],process_new_items:[4,2,1,""],random_state:[4,3,1,""],recommend:[4,2,1,""],run:[4,2,1,""],score_fn:[4,3,1,""],set_num_items_per_iter:[4,2,1,""],startup_and_train:[4,2,1,""],train:[4,2,1,""],users:[4,3,1,""],users_hat:[4,3,1,""]},"models.social":{SocialFiltering:[4,1,1,""]},"models.social.SocialFiltering":{process_new_items:[4,2,1,""]},"trecs.random.generators":{Generator:[5,1,1,""],SocialGraphGenerator:[5,1,1,""]},"trecs.random.generators.SocialGraphGenerator":{generate_random_graph:[5,2,1,""]},components:{base_components:[2,0,0,"-"],items:[2,0,0,"-"],socialgraph:[2,0,0,"-"],users:[2,0,0,"-"]},models:{bass:[4,0,0,"-"],content:[4,0,0,"-"],popularity:[4,0,0,"-"],recommender:[4,0,0,"-"],social:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"100":4,"1000":[4,5],"1000x1000":5,"100x200":4,"1200":4,"1250":4,"200":4,"2000":4,"2018":2,"2020":[1,2],"2376":2,"300":4,"500":4,"5000":4,"9739":2,"abstract":[2,3,4],"case":[2,3,4,5],"class":[2,4,5,6],"default":[2,3,4,5],"float":[2,3,4],"function":[2,3,4],"import":[1,5],"int":[2,3,4,5],"new":[0,1,2,3,4],"return":[2,3,4,5],"static":5,"true":[2,3,4],"try":0,"while":4,For:[2,4],That:2,The:[1,2,3,4],These:2,Useful:3,With:4,_old_histogram:3,_old_infection_st:3,_system_st:[2,4],about:[0,2,3,4],accept:2,accord:2,accur:3,across:2,activ:1,actual:[2,3,4],actual_item_represent:4,actual_user_profil:2,actual_user_represent:4,actual_user_scor:2,actualuserprofil:2,add:[1,2],add_friend:[2,4],add_new_item_indic:4,add_state_vari:2,added:[2,3,4],adding:[1,2],addit:4,adjac:[2,4,5],advanc:1,affect:2,after:[1,2,4],again:4,agent:[2,3],aggreg:3,all:[1,2,3,4],allow:[2,4],alpha:2,alreadi:[2,4],also:[1,4,5],alwai:4,among:3,anderson:3,ani:[0,2,4,5],anoth:2,anyth:2,api:[3,5],append:2,appli:4,arbitrari:2,arg:[2,4,5],argmax:2,argument:[3,4],around:5,arrai:[2,3,4],array_lik:[2,3,4],asap:0,assign:[2,4],assum:[2,3,4],assumpt:2,attent:2,attention_exp:2,attribut:[2,4],automat:1,avail:[1,4],avoid:1,base:[4,6],base_compon:2,basecompon:2,baseobserv:[2,3],baserecommend:[2,3,6],basic:2,bass:6,bassmodel:[3,4],been:[1,3,4],befor:[2,3],behavior:2,being:4,below:[1,4],beta:2,between:[1,2,3,4],beyond:4,bidirect:2,binari:[4,6],binarysocialgraph:[2,4],binomi:[4,5],bit_gener:5,bitgener:5,block:2,blog:1,bool:[2,3,4],both:2,branch:3,brief:1,build:[1,2],calc_dn_util:2,calcul:[2,3,4],call:[2,3,4],callabl:[2,4],can:[1,2,3,4,5],candid:[2,4],cannot:4,cdot:2,central:1,chanei:2,chang:[2,4,5],characterist:2,check:[0,1],choic:[2,4],choos:[2,4],choose_interleaved_item:4,chosen:[2,4],clone:1,code:[0,1],com:[1,2],combin:4,come:[1,3,4],command:1,common:4,commun:4,compat:2,complet:1,compon:[0,3,4,6],comput:[2,4],compute_user_scor:2,concept:[1,2,3],concret:[2,4],conda:1,configur:1,conflict:1,connect:[2,4],consist:[3,4],constraint:2,consum:2,contact:4,contain:[2,4],content:[0,6],contentfilt:[1,4],context:2,contributor:1,control:2,copi:[2,3],correct:0,correspond:2,could:4,creat:[1,2,4],create_and_process_item:4,creator:4,current:[1,2,3,4],custom:4,customiz:4,data:3,decemb:1,decid:2,defin:[2,4],definit:2,degre:[2,5],delet:2,denot:3,depend:[1,2,4],descript:2,design:[2,3],detail:[0,1,3,4,5],determin:[2,4],develop:3,dict:[2,3,4],dictionari:2,differ:[2,3],diffus:6,diffusion_tre:3,diffusiontreemeasur:3,dimens:[2,4],direct:2,directli:2,directori:1,disabl:[2,3,4],diseas:2,distinct:4,distr_typ:4,distribut:[1,2,4],divis:2,dnuser:2,doc:0,docstr:2,document:[3,4,5],doe:[2,4],done:4,dot:4,draft:1,draw:4,draw_diffusion_tre:4,draw_tre:3,drawn:4,drift:2,durat:3,dure:4,dynam:2,each:[2,3,4],easi:0,either:[2,4],element:[2,3,4],elig:4,elucherini:1,enabl:[2,3,4],encapsul:2,encourag:0,engag:4,enhanc:2,ensur:[1,3,4],enter:2,entri:4,environ:1,eps:2,equal:4,equat:2,equival:[3,5],error:[2,3,4],especi:0,essenti:2,establish:2,eta:2,etc:0,evalu:3,everi:[2,4],exact:2,exampl:[0,2,4,5],except:2,exclud:4,exist:2,expand:4,expect:2,expir:4,explain:0,explor:[1,4],expon:2,express:2,extend:3,extens:1,factor:4,fals:[2,3,4],featur:0,feel:0,few:3,filter:6,first:[0,1,4],fit:2,fix:0,flag:4,follow:[1,2,4],following_index:2,form:[0,2],found:1,foundat:4,frac:2,free:0,friend:2,from:[1,2,3,4,5],fromndarrai:2,gain:2,gather:4,gener:[0,2,4,6],generate_random_graph:5,generate_recommend:4,get:[2,4],get_actual_user_scor:2,get_component_st:2,get_measur:[1,3,4],get_observ:[2,3],get_structural_vir:[3,4],get_system_st:4,get_timestep:[2,3],get_user_feedback:2,git:1,github:[0,1,2],given:[2,3,4],glimcher:2,goe:2,goel:3,good:2,graph:[3,4,6],graph_typ:5,greater:2,guarante:4,guid:1,guidelin:1,had:4,has:[1,3,4],have:[1,2,4],help:[1,2,4],here:[1,2],histogram:3,histori:[2,4],hofman:3,homogen:6,homogeneitymeasur:3,how:[1,3,4],http:[1,2],i_u:2,ignor:4,implement:[2,3,4],improv:0,includ:[2,4,5],inclus:4,increas:4,independ:[2,4],index:[0,2,3],indic:[2,3,4],indirectli:4,individu:4,infect:[3,4],infection_prob:4,infection_st:[3,4],infection_threshold:4,infectionst:4,infectionthreshold:4,inform:[2,3,4],inherit:[2,3,4],init_valu:[2,3],initi:[2,3,4],initialize_user_scor:4,inner_product:[2,4],input:[2,4],input_arrai:[2,4],insert:4,instal:0,instanc:4,instanti:[2,4],instruct:1,integ:[4,5],interact:[2,4,6],interact_with_item:2,interaction_histogram:3,interactionmeasur:3,interactions_:2,interactions_u:2,interest:[1,2,3],interfac:3,interleav:4,interleaved_item:4,interleaving_fn:4,intern:[2,4],interpret:4,introduct:1,involv:2,issu:0,item:[3,4,6],item_attribut:[2,4],item_indic:4,item_represent:4,items_hat:4,items_shown:[2,4],iter:4,its:[1,2,4],itself:[3,4],jupyt:1,just:[2,4],keep:[3,4],kept:[3,4],keyword:3,knowledg:4,known:4,kwarg:[2,3,4,5],label:3,law:4,learn:1,length:3,librari:[3,4],like:4,likewis:4,limit:0,link:2,list:[2,3,4],load:[3,4],local:1,louie:2,maco:1,made:4,mai:4,main:1,mainli:2,make:2,manag:[1,2],mani:3,match:4,math:2,mathbf:2,mathrm:2,matplotlib:4,matric:5,matrix:[2,3,4,5],maxim:4,maximum:2,mean:[3,4],measur:[1,2,4,6],measure_cont:4,measurement_histori:3,mechan:2,method:[2,4],metric:[0,1,2,4,6],might:1,mileston:0,minim:5,mixin:2,mode:[2,3,4],model:[0,1,2,3,6],modul:[0,2,3,4],monitor:[2,4],more:[1,2,3,4,5],mse:6,msemeasur:3,much:0,multivari:2,must:[3,4],name:[1,2,3],ndarrai:[2,3,4,5],need:[1,2],network:[2,4,5],networkx:[3,5],neurobiolog:2,neuroecon:2,never:2,new_item:[2,4],no_new_item:4,node:[3,5],non:[2,3],none:[2,3,4,5],normal:2,normalize_valu:2,normed_valu:2,note:[1,2,3,4,5],notebook:1,now:4,num:5,num_attribut:4,num_choic:2,num_infect:3,num_item:4,num_items_per_it:[2,4],num_new_item:4,num_us:[2,4],number:[2,3,4,6],number_of_item:2,number_of_us:2,numpi:[2,3,4,5],object:[2,4],observ:[2,3],observable_typ:2,older:1,omega:2,onc:[2,4],one:[2,3,4],ones:4,onli:[1,3,4],onlin:[3,4],onto:3,oper:2,option:[2,3,4,5],order:[2,4],origin:2,other:[2,4,5],otherwis:4,our:[2,4],out:[1,4],outcom:3,over:[2,3],overrid:4,overview:1,own:[1,4],packag:1,page:0,pair:[2,4],panda:1,param:4,paramet:[2,3,4,5],parent:4,particular:4,pass:[3,4],past:4,pattern:[2,3],per:[3,4],perceiv:4,percentag:4,perfect:4,perform:2,period:4,perturb:2,pip:1,plan:1,pleas:[0,1,3,4,5],plot:3,popular:6,popularityrecommend:4,possibl:[0,2],post:1,power:4,power_dist:4,powerlaw:4,pre:[3,4],predict:[2,3,4],predicted_scor:4,predictedscor:[2,4],predicteduserprofil:[2,4],prefer:[2,3,4],present:4,previou:[3,4],previous:4,probabilistic_recommend:4,probabl:4,process:4,process_new_item:4,product:4,profil:[2,4],programm:3,propag:4,proportion:4,provid:4,pull:0,py37:1,pylint:1,python:1,quantiti:3,question:0,quick:1,rais:[2,5],randint:4,random:[0,2,4,6],random_graph:5,random_items_per_it:4,random_regular_graph:5,random_st:4,randomli:[2,4],rang:2,rank:[2,4],rather:4,real:[2,3,4],rec:0,receiv:[3,4],recent:4,recommend:[1,2,3,6],record:[2,3,4],record_base_st:4,recsi:0,refer:[0,3,4,5],regard:0,regardless:2,register_observ:2,relat:2,relationship:[2,4],remov:2,remove_friend:[2,4],repeat:[2,4],repeat_interact:2,repeated_item:4,report:0,repres:[2,3,4],represent:[2,4],request:0,requir:[2,5],resourc:1,respond:0,result:4,retrain:4,rotat:2,round:0,row:2,rrg:5,run:[1,4],sai:4,same:[2,4],sampl:2,sample_from_error_dist:2,scienc:2,scientif:1,scipi:1,score:[2,3,4],score_fn:[2,4],score_new_item:2,search:0,second:[2,4],see:[2,3,4,5],seed:[2,4,5],select:2,send:0,serv:4,set:[2,4],set_num_items_per_it:4,set_score_funct:2,shape:[2,4],should:[1,2,3,4],shown:4,sigma:2,signal:0,similar:[2,4],simpl:3,simpli:4,simul:[3,4],sinc:4,singl:2,size:[2,4,5],social:6,socialfilt:4,socialgraph:2,socialgraphgener:5,sociotechn:1,sole:4,some:[2,3],soon:1,sourc:[2,3,4,5],specif:[2,3],specifi:2,spread:[3,4],squar:3,stage:2,start:[1,3,4],startup:4,startup_and_train:4,state:[2,3,4],state_histori:[2,4],step:[3,4],store:[2,3,4],store_st:2,str:[2,3,4],structur:[4,6],structuralvir:3,studi:4,subclass:2,suit:2,sum_n:2,supplement:1,support:[1,2,4],suppos:2,suscept:4,system:[1,2,3,4,5],system_st:4,systemstatemodul:[2,4],take:[2,4],taken:2,term:2,termin:1,test:1,text:2,textbf:2,than:[2,4],thank:1,thei:[2,4],them:2,therefor:[2,4],thi:[1,2,3,4,5],thin:5,threshold:4,through:0,thrown:4,time:[2,3,4],timestep:[1,2,3,4],todo:4,tool:0,top:[1,4],total:3,toward:2,track:[2,3,4],train:[2,4],train_between_step:4,transform:2,trec:[1,4,5],tree:[4,6],tupl:2,tutori:0,two:2,type:[2,3,4,5],typeerror:2,ubuntu:1,unawar:4,under:4,underli:[2,4],unfollow:[2,4],unidirect:2,uniformli:4,unknown:[2,4],unless:2,unregister_observ:2,uoft:2,updat:[2,3,4],update_profil:2,usag:0,use:[0,1,5],used:[1,2,3,4],useful:[0,1,4],user1_index:2,user2_index:2,user:[3,4,5,6],user_index:2,user_item_scor:2,user_profil:[2,4],user_represent:4,user_vector:2,users_hat:[2,4],uses:4,using:[1,3,4],util:2,v_i:2,v_n:2,valid:4,valu:[2,3,4],valuat:2,valueerror:[2,5],vari:2,variabl:2,vary_random_items_per_it:4,vector:[2,4],verbos:[2,3,4],version:1,viral:[4,6],virtual:1,wai:2,want:4,watt:3,webb:2,welcom:0,well:[2,3,4],were:4,what:0,when:[2,3,4],where:[2,3,4],whether:[2,3,4],which:[2,3,4],whole:2,within:4,won:4,would:0,wrapper:5,wrong:2,yet:4,you:[0,1],your:1,z_i:2,zero:[2,4]},titles:["Simulator Documentation","T-RECS (Tool for RecSys Simulation)","Components","Metrics","Models","Random Generators","Reference"],titleterms:{"class":3,base:[2,3],baserecommend:4,bass:4,binari:2,bug:0,compon:2,content:4,contribut:[0,1],diffus:[3,4],document:[0,1],exampl:1,feedback:0,filter:4,gener:5,graph:[2,5],homogen:3,inconsist:0,indic:0,instal:1,interact:3,item:2,known:0,measur:3,metric:3,model:4,mse:3,number:5,popular:4,random:5,rec:1,recommend:4,recsi:1,refer:6,simul:[0,1],social:[2,4,5],structur:3,tabl:0,todo:2,tool:1,tree:3,tutori:1,usag:1,user:2,viral:3}})