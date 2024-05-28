clear
clc
angle=[];                                                                              
magnitude=[];                                                                          
TotalDC=[];                                                                            
tic                                                                                    
define_constants; 
grid_name = input('Enter powergrid IEEE24, IEEE39, IEEE118, UK:  ', 's');
mpctry_file = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\13_Power_system\' grid_name '\System'];
demand_file = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\13_Power_system\' grid_name '\hourlyDemandBus.mat'];
save_folder = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\code\dataset\' grid_name '\raw'];

mpctry = loadcase(mpctry_file);
load(demand_file);
n_instances = size(hourlyDemandBus,2);
index_loads = find(mpctry.bus(:,3)~=0); 

nodes = size(mpctry.bus,1);
[cnt_unique, index_gen] = hist(mpctry.gen(:,1),unique(mpctry.gen(:,1)));
num_gen = zeros(nodes,1);
num_gen(index_gen)=cnt_unique;
flag_load = zeros(nodes,1);
flag_load(index_loads)=1;
% demand_PD = createunif_demand(nodes, index_loads, n_instances);
% demand_QD = createunif_demand(nodes, index_loads, n_instances);
iteration= 0;
for j=1:n_instances                                                                           
    for i=1:nodes                                                                             
        mpctry.bus(i, PD)= hourlyDemandBus(i,j);                                               
    end                                                                                    
                                                           
    [case24new,success]=runpf(mpctry);     
     if success==0
         continue
     else
         iteration=iteration+1;
     end
     angle =case24new.bus(:,9);                                                             
     mag = case24new.bus(:,8);
     Vr = zeros(nodes,1);
     Vi = zeros(nodes,1);
     Pg = zeros(nodes,1);
     Qg = zeros(nodes,1); 
     for k = 1:nodes
        [Vr(k), Vi(k)] = pol2cart(mag(k),angle(k));
     end
     G_index = case24new.gen(:,1);     
  
     Pg(G_index) = case24new.gen(:,2);  
     Qg(G_index) = case24new.gen(:,3);  
     
     X{iteration} = [mpctry.bus(:, PD), mpctry.bus(:, QD), Pg, Qg, flag_load,num_gen];                                           
     %result_GenActivePower=[accumarray(G_index,GenActivePower);0];                         
                                                     
     %result_GenReactivePower=[accumarray(G_index,GenReactivePower);0];   

     Y_polar{iteration} = [mag, angle];
     Y_rect{iteration} = [Vr, Vi];

     %LoadActivePower=case24new.bus(:,3);                                                   
     %LoadReactivePower=case24new.bus(:,4);                                                 
     %NetREALPower=[abs(result_GenActivePower-LoadActivePower)];                            
     %NetReactivePower=[abs(result_GenReactivePower-LoadReactivePower)];                    
                                                                                             
end   
toc;
toc-tic
edge_index_noloops=[case24new.branch(:,1),case24new.branch(:,2)];
save(fullfile(save_folder, 'X.mat'), 'X', '-v7.3');
save(fullfile(save_folder, 'Y_polar.mat'), 'Y_polar', '-v7.3');
save(fullfile(save_folder, 'Y_rect.mat'), 'Y_rect', '-v7.3');
%save('edge_index.mat','edge_index','-v7.3')
          
[Ybus, Yf, Yt] = makeYbus(mpctry.baseMVA, mpctry.bus, mpctry.branch);
[i,j,s] = find(Ybus);
edge_index = [i,j];
G = real(s);
% save 'G_try.mat' G
B = imag(s);
% save 'B_try.mat' B
edge_attr = [G, B];
save(fullfile(save_folder, 'edge_index.mat'), 'edge_index');
save(fullfile(save_folder, 'edge_attr.mat'), 'edge_attr');

                             