clear
clc
angle=[];                                                                              
magnitude=[];                                                                          
TotalDC=[];                                                                            
tic                                                                                    
define_constants;                                                                      
grid_name = input('Enter powergrid IEEE24, IEEE39, IEEE118, UK:  ', 's');
mpctry_file = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\13_Power_system\' grid_name '\System'];
demand_file = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\13_Power_system\' grid_name '\hourlyDemandBusnew.mat'];
save_folder = ['C:\Users\avarbella\OneDrive - ETH Zurich\Documents\01_GraphGym\PowerGraph-masterdataset-node\code\dataset\' lower(grid_name) '\' lower(grid_name) '\raw'];

mpctry = loadcase(mpctry_file);
load(demand_file);
n_instances = size(hourlyDemandBusnew,2);
index_loads = find(mpctry.bus(:,3)~=0); 

nodes = size(mpctry.bus,1);
[cnt_unique, index_gen] = hist(mpctry.gen(:,1),unique(mpctry.gen(:,1)));
num_gen = zeros(nodes,1);
num_gen(index_gen)=cnt_unique;
flag_load = zeros(nodes,1);
flag_load(index_loads)=1;
% demand_PD = createunif_demand(nodes, index_loads, n_instances);
% demand_QD = createunif_demand(nodes, index_loads, n_instances);
iteration= 1;
bus_type = mpctry.bus(:,2);
indices_PQ = find(bus_type == 1);
indices_PV = find(bus_type == 2);
indices_slack = find(bus_type == 3);
for j=1:n_instances
    %%%%%%%%%%%%% SET DEMAND %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:nodes                                                                             
        mpctry.bus(i, PD)= hourlyDemandBusnew(i,j);                                               
    end  
    Pl = mpctry.bus(indices_PQ, PD);
    Pl_gen = mpctry.bus(indices_PV, PD);
    Pl_slack = mpctry.bus(indices_slack, PD);
    Ql_gen = mpctry.bus(indices_PV, QD);
    Ql_slack = mpctry.bus(indices_slack, QD);
    Ql = mpctry.bus(indices_PQ, QD);
    X{iteration} = zeros(nodes,4);
     %%%%%%%%%%%%% SET NODE f matrix and OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%
    X{iteration}(indices_PQ,:) = [-Pl, -Ql, zeros(length(indices_PQ),1), bus_type(indices_PQ)];                                           
    X{iteration}(indices_PV,:) = [-Pl_gen, -Ql_gen, zeros(length(indices_PV),1), bus_type(indices_PV)];                                           
    X{iteration}(indices_slack,:) = [-Pl_slack , -Ql_slack, zeros(length(indices_slack),1), bus_type(indices_slack)];        
    %%%%%%%%%%%%% RUN OPF    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [case24new,success]=runopf(mpctry);    
    if success==0
        continue
    else
        iteration=iteration+1;
    end    
    %%%%%%%%%%%%% PROCESS RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    angle =case24new.bus(:,9);                                                             
    mag = case24new.bus(:,8);
    Vr = zeros(nodes,1);
    Vi = zeros(nodes,1);
    Pg = zeros(nodes,1);
    Qg = zeros(nodes,1); 
    for k = 1:nodes
        [Vr(k), Vi(k)] = pol2cart(mag(k),angle(k));
    end
    %%%%%%%%%%%%% GENERATORS      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index_gen = unique(case24new.gen(:,1));

    %%%%%%%%%%%%% tot POWER Generated per bus with generators  %%%%%%%%%%%%
     Pg_bus = zeros(length(index_gen), 1);
     Qg_bus = zeros(length(index_gen), 1);
     for i = 1:length(index_gen)
        index = index_gen(i);
        rows = case24new.gen(:,1) == index;      
        summed_value_p = sum(case24new.gen(rows,2));
        summed_value_q = sum(case24new.gen(rows,3));
        Pg_bus(i, :) = summed_value_p;
        Qg_bus(i, :) = summed_value_q;
     end          
     Pg(index_gen,:) = Pg_bus;  
     Qg(index_gen,:) = Qg_bus; 
     %%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P        Q         V          T %%%%%%%
                                 
     Y_polar{iteration-1}(indices_PQ,:) = [zeros(length(indices_PQ),1), zeros(length(indices_PQ),1), mag(indices_PQ), angle(indices_PQ)];
     Y_polar{iteration-1}(indices_PV,:) = [Pg(indices_PV), Qg(indices_PV), mag(indices_PV), angle(indices_PV)];
     Y_polar{iteration-1}(indices_slack,:) = [Pg(indices_slack), Qg(indices_slack), mag(indices_slack), angle(indices_slack)];
%%     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% POWER FLOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%% set generators with opf results %%%%%%%%%%%
     mpctry.gen(:,2) = case24new.gen(:,2); 
     mpctry.gen(:,3) = case24new.gen(:,3); 

     %%%%%%%%%%%%% PROCESS INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     angle =case24new.bus(:,9);                                                             
     mag = case24new.bus(:,8);
     Vr = zeros(nodes,1);
     Vi = zeros(nodes,1);
     Pg = zeros(nodes,1);
     Qg = zeros(nodes,1); 
     for i = 1:length(index_gen)
        index = index_gen(i);
        rows = case24new.gen(:,1) == index;
        summed_value_p = sum(case24new.gen(rows,2));
        summed_value_q = sum(case24new.gen(rows,3));
        Pg_bus(i, :) = summed_value_p;
        Qg_bus(i, :) = summed_value_q;
     end
     Pg(index_gen,:) = Pg_bus; 
     Qg(index_gen,:) = Qg_bus; 
     for k = 1:nodes
        [Vr(k), Vi(k)] = pol2cart(mag(k),angle(k));
     end
     Pl = mpctry.bus(indices_PQ, PD);
     Pl_gen = mpctry.bus(indices_PV, PD);
     Ql_gen = mpctry.bus(indices_PV, QD);
     Ql = mpctry.bus(indices_PQ, QD);
     Xpf{iteration-1} = zeros(nodes,4);
     %%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P        Q         V          T %%%%%%%
     Xpf{iteration-1}(indices_PQ,:) = [-Pl, -Ql, zeros(length(indices_PQ),1), bus_type(indices_PQ)];                                           
     Xpf{iteration-1}(indices_PV,:) = [-Pl_gen+Pg(indices_PV), -Ql_gen, mag(indices_PV) , bus_type(indices_PV)];                                           
     Xpf{iteration-1}(indices_slack,:) = [0 , 0, mag(indices_slack), bus_type(indices_slack)];  

     %%%%%%%%%%%%%%%%%%%%%%%%%% RUN POWER FLOW  %%%%%%%%%%%%%%%%%%%%%%%%%%%
     [case24new,success]=runpf(mpctry);   
    %%%%%%%%%%%%% GENERATORS      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Pg = zeros(nodes,1);
     Qg = zeros(nodes,1);  
     Pg_bus = zeros(length(index_gen), 1);
     Qg_bus = zeros(length(index_gen), 1);
     for i = 1:length(index_gen)
        index = index_gen(i);
        rows = case24new.gen(:,1) == index;
        summed_value_p = sum(case24new.gen(rows,2));
        summed_value_q = sum(case24new.gen(rows,3));
        Pg_bus(i, :) = summed_value_p;
        Qg_bus(i, :) = summed_value_q;
     end
     Pg(index_gen,:) = Pg_bus; 
     Qg(index_gen,:) = Qg_bus; 
     %%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P        Q         V          T %%%%%%%
     Y_polarpf{iteration-1}(indices_PQ,:) = [zeros(length(indices_PQ),1), zeros(length(indices_PQ),1), mag(indices_PQ), angle(indices_PQ)];
     Y_polarpf{iteration-1}(indices_PV,:) = [zeros(length(indices_PV),1), Qg(indices_PV),zeros(length(indices_PV),1), angle(indices_PV)];
     Y_polarpf{iteration-1}(indices_slack,:) = [Pg(indices_slack), Qg(indices_slack), 0, 0];
    
               
                                                                                             
end   
toc;
toc-tic
[Ybus, Yf, Yt] = makeYbus(mpctry.baseMVA, mpctry.bus, mpctry.branch);
[i,j,s] = find(Ybus);
edge_index = [i,j];
G = real(s);
% save 'G_try.mat' G
B = imag(s);
% save 'B_try.mat' B
edge_attr = [G, B];
%%%%%%%%%%%%%%%%%%%%%%%%%%% SAVE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
edge_index_noloops=[case24new.branch(:,1),case24new.branch(:,2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%% OPF  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save(fullfile(save_folder, 'Xopf.mat'), 'X', '-v7.3');
save(fullfile(save_folder, 'Y_polar_opf.mat'), 'Y_polar', '-v7.3');
save(fullfile(save_folder, 'edge_index_opf.mat'), 'edge_index');
save(fullfile(save_folder, 'edge_attr_opf.mat'), 'edge_attr');
%%%%%%%%%%%%%%%%%%%%%%%%%%% PF  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save(fullfile(save_folder, 'X.mat'), 'Xpf', '-v7.3');
save(fullfile(save_folder, 'Y_polar.mat'), 'Y_polarpf', '-v7.3');
save(fullfile(save_folder, 'edge_index.mat'), 'edge_index');
save(fullfile(save_folder, 'edge_attr.mat'), 'edge_attr');

                             