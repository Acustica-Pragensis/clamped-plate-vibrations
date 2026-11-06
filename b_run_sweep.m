clear;
clc;

% ------------------------------ configuration -------------------------------
num_modes      = 17;
elems_per_side = 61;

MODEL_ROOT = 'models';
INVENTORY  = 'em_inventory.csv';

timestamp = string(datetime('now'), 'yyyyMMdd_HHmmss');
fprintf('run started %s\n', timestamp);

% -------------------------------- splits ------------------------------------
ba_train = [1.03, 1.10, 1.18, 1.40, 1.50, 1.70, 1.80];
a_train  = [3.0e-4, 6.0e-4, 9.0e-4, 12.0e-4, 15.0e-4];
h_train  = [4.0e-6, 7.0e-6, 10.0e-6, 13.0e-6];

ba_test  = [1.30, 1.60, 1.90, 2.00];
a_test   = [4e-4, 7e-4, 10e-4, 13e-4];
h_test   = [5e-6, 8e-6, 12e-6];

split_name = {'train', 'test'};
split_csv  = {'ef_comsol_train.csv', 'ef_comsol_test.csv'};
split_ba   = {ba_train, ba_test};
split_a    = {a_train,  a_test};
split_h    = {h_train,  h_test};

n_split = numel(split_name);

if ~exist(MODEL_ROOT, 'dir'); mkdir(MODEL_ROOT); end

% ------------------------- inventory (preallocated) -------------------------
n_total = sum(cellfun(@(x, y, z) numel(x)*numel(y)*numel(z), ...
                      split_ba, split_a, split_h));

inv_split = cell(n_total, 1);
inv_ba    = nan(n_total, 1);
inv_a     = nan(n_total, 1);
inv_h     = nan(n_total, 1);
inv_em    = cell(n_total, 1);
n_inv     = 0;

% -------------------------------- main sweep --------------------------------
n_done = 0;
t_all  = tic;

for s = 1:n_split

    sp_name = split_name{s};
    sp_csv  = split_csv{s};
    sp_ba   = split_ba{s};
    sp_a    = split_a{s};
    sp_h    = split_h{s};

    n_geo = numel(sp_ba) * numel(sp_a) * numel(sp_h);
    fprintf('\n===== %s set: %d geometries =====\n', sp_name, n_geo);

    mph_dir = fullfile(MODEL_ROOT, sp_name);
    if ~exist(mph_dir, 'dir'); mkdir(mph_dir); end

    results = nan(n_geo, 3 + num_modes);
    i = 0;

    for ba = sp_ba
        for a = sp_a
            for h = sp_h

                i      = i + 1;
                n_done = n_done + 1;
                fprintf('[%3d/%3d] %-5s ba=%1.2f a=%.2e h=%.2e ... ', ...
                        n_done, n_total, sp_name, ba, a, h);
                t_geo = tic;

                base     = sprintf('%s_ba%1.2f_a%1.1e_h%1.1e', sp_name, ba, a, h);
                mph_file = fullfile(mph_dir, [base, '.mph']);

                mobj  = c_rectangle(a, ba, h, num_modes, elems_per_side);
                mphsave(mobj, mph_file);
                freqs = mphglobal(mobj, 'freq');

                if numel(freqs) ~= num_modes
                    warning('expected %d modes, got %d (%s)', ...
                            num_modes, numel(freqs), base);
                    freqs = freqs(:)';
                    nf = min(numel(freqs), num_modes);
                    frow = nan(1, num_modes);
                    frow(1:nf) = freqs(1:nf);
                else
                    frow = freqs(:)';
                end
                results(i, :) = [ba, a, h, frow];

                n_inv            = n_inv + 1;
                inv_split{n_inv} = sp_name;
                inv_ba(n_inv)    = ba;
                inv_a(n_inv)     = a;
                inv_h(n_inv)     = h;
                inv_em{n_inv}    = sprintf('em_ba%1.2f_a%1.1e_h%1.1e_W', ba, a, h);

                % free the COMSOL model before the next geometry
                try
                    ModelUtil.clear(mobj);
                catch
                end

                fprintf('%.1f s\n', toc(t_geo));

            end % h

            % ----- save after each completed h_sweep -----
            write_ef_csv(sp_csv, results(1:i, :), num_modes);
            write_inventory(INVENTORY, inv_split(1:n_inv), inv_ba(1:n_inv), ...
                            inv_a(1:n_inv), inv_h(1:n_inv), inv_em(1:n_inv));
            fprintf('    saved %s (%d geometries), %s (%d rows)\n', ...
                    sp_csv, i, INVENTORY, n_inv);

        end % a
    end % ba

    % final write of the split
    write_ef_csv(sp_csv, results(1:i, :), num_modes);
    fprintf('===== %s set finished: %d geometries =====\n', sp_name, n_geo);

end % splits

fprintf('\nall done in %.1f min\n', toc(t_all) / 60);

% ------------------------------ local functions -----------------------------
function write_ef_csv(filename, rows, num_modes)
    headers = [{'ba'}, {'a'}, {'h'}, ...
               arrayfun(@(x) sprintf('f%d', x), 1:num_modes, ...
                        'UniformOutput', false)];
    T = array2table(rows, 'VariableNames', headers);
    writetable(T, filename);
end

function write_inventory(filename, split, ba, a, h, em)
    T = table(split, ba, a, h, em, ...
              'VariableNames', {'split', 'ba', 'a', 'h', 'em_csv'});
    writetable(T, filename);
end