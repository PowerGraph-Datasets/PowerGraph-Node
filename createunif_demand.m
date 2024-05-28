function [demand_bus] = createunif_demand(number_bus, index_loads, n_instances)
% creates a hourlyDemandBus.mat, from a uniform distribution
% inputs: -number of bus
%           -index_loads: at what bus are load present
%           - number of hours (instance to generate)

demand_bus = zeros([number_bus, n_instances]);
demand_bus_load_only = unifrnd(0,1,[length(index_loads), n_instances]);
for i = 1:length(index_loads)
  demand_bus(index_loads(i), :) = demand_bus_load_only(i,:); 
end
