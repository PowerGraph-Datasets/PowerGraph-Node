import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Specify the path to your Excel file
models_path = 'C:\\Users\\avarbella\\OneDrive - ETH Zurich\\Documents\\01_GraphGym\\PowerGraph-masterdataset-node\\code\\modelbig\\'
sheet_name_d = 'Data'

path_best_results = {'ieee24': {'node': 'summaryieee24_gat_node_2l_32h_100s.xlsx', 'nodeopf': 'summaryieee24_gat_nodeopf_1l_32h_100s.xlsx'},
                     'ieee39': {'node': 'summaryieee39_gat_node_2l_32h_300s.xlsx', 'nodeopf': 'summaryieee39_transformer_nodeopf_3l_8h_0s.xlsx'},
                     'uk': {'node': 'summaryuk_transformer_node_3l_16h_300s.xlsx', 'nodeopf': 'summaryuk_gat_nodeopf_2l_32h_700s.xlsx'},
                     'ieee118': {'node': 'summaryieee118_transformer_node_1l_16h_300s.xlsx', 'nodeopf': 'summaryieee118_transformer_nodeopf_1l_16h_100s.xlsx'}}

n_buses = [24, 39, 29, 118]

mse_ac_node = []
mse_bd_node = []
mse_pg_node = []
mse_qg_node = []
mse_ac_nodeopf = []
mse_bd_nodeopf = []
mse_pg_nodeopf = []
mse_qg_nodeopf = []
i = 0
for grid, paths in path_best_results.items():
    n_bus = n_buses[i]
    i+= 1
    for task, path in paths.items():
        if task == 'node':
            plt.figure(0)
            best_excel_file_path = models_path + grid + '\\'+path
            workbookplot = openpyxl.load_workbook(best_excel_file_path)
            sheet = workbookplot[sheet_name_d]
    # Ge    t data from columns A, B, C, and D
            v_targ = [cell.value for cell in sheet['A']]
            t_targ = [cell.value for cell in sheet['B']]
            pg_targ = [cell.value for cell in sheet['C']]
            qg_targ = [cell.value for cell in sheet['D']]
            v_pred = [cell.value for cell in sheet['E']]
            t_pred = [cell.value for cell in sheet['F']]
            pg_pred = [cell.value for cell in sheet['G']]
            qg_pred = [cell.value for cell in sheet['H']]
            l = len(v_targ)
            mse_ac_node.append(sum(np.square(np.array(v_targ[1:]) - np.array(v_pred[1:])))/l/n_bus)
            mse_bd_node.append(sum(np.square(np.array(t_targ[1:]) - np.array(t_pred[1:])))/l/n_bus)
            mse_pg_node.append(sum(np.square(np.array(pg_targ[1:]) - np.array(pg_pred[1:])))/l/n_bus)
            mse_qg_node.append(sum(np.square(np.array(qg_targ[1:]) - np.array(qg_pred[1:])))/l/n_bus)



        elif task == 'nodeopf':
            plt.figure(0)
            best_excel_file_path = models_path + grid + '\\'+path
            workbookplot = openpyxl.load_workbook(best_excel_file_path)
            sheet = workbookplot[sheet_name_d]
            # Get data from columns A, B, C, and D
            v_targ = [cell.value for cell in sheet['A']]
            t_targ = [cell.value for cell in sheet['B']]
            pg_targ = [cell.value for cell in sheet['C']]
            qg_targ = [cell.value for cell in sheet['D']]
            v_pred = [cell.value for cell in sheet['E']]
            t_pred = [cell.value for cell in sheet['F']]
            pg_pred = [cell.value for cell in sheet['G']]
            qg_pred = [cell.value for cell in sheet['H']]
            l= len(v_targ)
            # Calculate MSE between columns
            mse_ac_nodeopf.append(sum(np.square(np.array(v_targ[1:]) - np.array(v_pred[1:])))/l/n_bus)
            mse_bd_nodeopf.append(sum(np.square(np.array(t_targ[1:]) - np.array(t_pred[1:])))/l/n_bus)
            mse_pg_nodeopf.append(sum(np.square(np.array(pg_targ[1:]) - np.array(pg_pred[1:])))/l/n_bus)
            mse_qg_nodeopf.append(sum(np.square(np.array(qg_targ[1:]) - np.array(qg_pred[1:])))/l/n_bus)


quantities = ['V [p.u.]', 'T [degree]', 'Pg [MW]', 'Qg [MVAr]']
data = {
    'quantity': quantities * 2,
    'Mean Absolute Error (MAE)': mse_ac_node + mse_ac_nodeopf,
    'dataset': ['mse_ac_node'] * len(quantities) + ['mse_ac_nodeopf'] * len(quantities)
}
df = pd.DataFrame(data)

# Define color palette
pal_node = sns.color_palette("Dark2", 4)  # 4 colors for 4 quantities
pal_nodeopf = sns.color_palette("Dark2", 4)  # 4 colors for 4 quantities

# Combine the data for easier plotting
grids = ['IEEE24', 'IEEE39', 'UK', 'IEEE118']
data_node = [mse_ac_node, mse_bd_node, mse_pg_node, mse_qg_node]
data_nodeopf = [mse_ac_nodeopf, mse_bd_nodeopf, mse_pg_nodeopf, mse_qg_nodeopf]

# Set the color palettes
pal_node = sns.color_palette("Dark2", 4)
pal_nodeopf = sns.color_palette("Dark2", 4)

# Plot for node
plt.figure(figsize=(10, 6))
for i, quantity in enumerate(quantities):
    plt.bar(np.arange(4) + i * 0.2, data_node[i], width=0.2, color=pal_node[i], label=grids[i])

plt.xlabel('Quantity', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.title('Power flow', fontsize=16)
plt.xticks(np.arange(4) + 0.3, ['V', 'T', 'Pg', 'Qg'], fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.show()

# Plot for nodeopf
plt.figure(figsize=(10, 6))
for i, quantity in enumerate(quantities):
    plt.bar(np.arange(4) + i * 0.2, data_nodeopf[i], width=0.2, color=pal_nodeopf[i], label=grids[i])

plt.xlabel('Quantity', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.title('Optimal power flow', fontsize=16)
plt.xticks(np.arange(4) + 0.3, ['V', 'T', 'Pg', 'Qg'], fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.show()

