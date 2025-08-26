function plot_data()
    fname = 'C:\Workspace\Thesis\Battery_SoC_Estimation\dataset\LG_HG2_processed\25degC\549_HPPC_processed.csv';

    % --- Read table, preferring original headers ---
    tbl = [];
    try
        % Newer MATLAB: preserve headers exactly as in the CSV
        opts = detectImportOptions(fname, 'VariableNamingRule','preserve');
        tbl  = readtable(fname, opts);
    catch
        % Fallback for older MATLAB (no VariableNamingRule)
        tbl  = readtable(fname);  % headers may be sanitized
    end

    % --- Pull required variables (robust to sanitized names) ---
    time = getVar(tbl, "Time [s]");
    voltage = getVar(tbl, "Voltage [V]");
    current = getVar(tbl, "Current [A]");
    temperature = getVar(tbl, "Temperature [degC]");
    soc = getVar(tbl, "SOC [-]");

    % ---- Plot ----
    tiledlayout(4,1, "TileSpacing","compact", "Padding","compact");

    nexttile;
    plot(time, voltage); grid on;
    xlabel('Time [s]'); ylabel('Voltage [V]'); title('Voltage vs Time');

    nexttile;
    plot(time, current); grid on;
    xlabel('Time [s]'); ylabel('Current [A]'); title('Current vs Time');

    nexttile;
    plot(time, temperature); grid on;
    xlabel('Time [s]'); ylabel('Temperature [°C]'); title('Temperature vs Time');

    nexttile;
    plot(time, soc); grid on;
    xlabel('Time [s]'); ylabel('SOC [-]'); title('State of Charge vs Time');

    sgtitle('HPPC Processed Data');
end

function v = getVar(tbl, originalName)
    % Try direct access first (works if headers were preserved)
    if any(strcmp(tbl.Properties.VariableNames, originalName))
        v = tbl.(originalName);
        return;
    end

    % Otherwise, the table likely sanitized names. Use VariableDescriptions
    % to find which sanitized name corresponds to the original header.
    VD = tbl.Properties.VariableDescriptions;

    % If VariableDescriptions is empty or all empty, try a looser match
    if isempty(VD) || all(cellfun(@(c) isempty(c) || all(isspace(c)), VD))
        % Last resort: heuristic partial match on sanitized names
        % (e.g., "Time_s_" for "Time [s]")
        candidates = contains(lower(tbl.Properties.VariableNames), lower(regexprep(originalName,'[^A-Za-z0-9]','')));
        idx = find(candidates, 1);
        if ~isempty(idx)
            v = tbl.(tbl.Properties.VariableNames{idx});
            warning('Using heuristic match for "%s" -> "%s".', originalName, tbl.Properties.VariableNames{idx});
            return;
        end
        error('Could not find variable "%s" in the table.', originalName);
    end

    % Map original -> sanitized via VariableDescriptions
    orig = string(VD);
    sanitized = string(tbl.Properties.VariableNames);
    idx = find(orig == originalName, 1);

    if isempty(idx)
        % Sometimes orig may contain trimmed or slightly altered text; try contains
        idx = find(contains(orig, originalName), 1);
    end

    if isempty(idx)
        error('Could not find variable "%s" (check header names in the CSV).', originalName);
    end

    v = tbl.(sanitized(idx));
end
