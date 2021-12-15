%GET_TOTAL_RSS Calculates the Received Signal Strength (RSS) in dBm from
% a CSI struct.
%
% (c) 2011 Daniel Halperin <dhalperi@cs.washington.edu>
%
function ret = get_total_rss(rssi_a,rssi_b,rssi_c,agc)
    error(nargchk(1,4,nargin));

    % Careful here: rssis could be zero
    rssi_mag = 0;
    if rssi_a ~= 0
        rssi_mag = rssi_mag + dbinv(rssi_a);
    end
    if rssi_b ~= 0
        rssi_mag = rssi_mag + dbinv(rssi_b);
    end
    if rssi_c ~= 0
        rssi_mag = rssi_mag + dbinv(rssi_c);
    end
    
    ret = db(rssi_mag, 'pow') - 44 - agc;
end


function rett = dbinv(x)
    rett = 10.^(x/10);
end