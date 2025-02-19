function convert_Bennu_to_json(json_filename,sourceURL,sourceDescription,particles)

if particles == 1
    Bennu_grav_20_particles;
    VALSp = VALS;
    NAMESp = NAMES;
    nump = length(NAMESp);
    DEGREEp = DEGREE;
end

Bennu_grav_shape_16x16;
num = length(NAMES);

if particles == 1
    for ii = 1:nump
        if ~isempty(strfind(NAMESp{ii},'GM2101955'))
            break
        end
    end
    mu = VALSp(ii);
else
    for ii = 1:num
        if ~isempty(strfind(NAMES{ii},'GM2101955'))
            break
        end
    end
    mu = VALS(ii);
end

fid_out = fopen(json_filename,'w');
fprintf(fid_out,'{\n');
fprintf(fid_out,['    "BodyName":"' REF_BODY '",\n']);
fprintf(fid_out,['    "ReferenceRadius":' num2str(REF_RADIUS,6) ',\n']);
fprintf(fid_out,['    "GravitationalParameter":' num2str(mu,12) ',\n']);
if NORMALIZED == 0
    fprintf(fid_out,'    "Normalized":false,\n');
elseif NORMALIZED == 1
    fprintf(fid_out,'    "Normalized":true,\n');
else
    error('Unknown normalization! Panic!')
end
fprintf(fid_out,['    "MaxDegree":' num2str(DEGREE,'%d') ',\n']);
fprintf(fid_out,'    "Coefficients":[\n');

% Coefficients start with J2, assume C[0,0] = 1 and degree 1 terms are 0
fprintf(fid_out,'        {\n');
fprintf(fid_out,['            "degree":' num2str(0,'%d') ',\n']);
fprintf(fid_out,['            "order":' num2str(0,'%d') ',\n']);
fprintf(fid_out,['            "Cnm":' num2str(1.0,'%16.15E') ',\n']);
fprintf(fid_out,['            "Snm":' num2str(0.0,'%16.15E') '\n']);
fprintf(fid_out,'        },\n');
fprintf(fid_out,'        {\n');
fprintf(fid_out,['            "degree":' num2str(1,'%d') ',\n']);
fprintf(fid_out,['            "order":' num2str(0,'%d') ',\n']);
fprintf(fid_out,['            "Cnm":' num2str(0.0,'%16.15E') ',\n']);
fprintf(fid_out,['            "Snm":' num2str(0.0,'%16.15E') '\n']);
fprintf(fid_out,'        },\n');
fprintf(fid_out,'        {\n');
fprintf(fid_out,['            "degree":' num2str(1,'%d') ',\n']);
fprintf(fid_out,['            "order":' num2str(1,'%d') ',\n']);
fprintf(fid_out,['            "Cnm":' num2str(0.0,'%16.15E') ',\n']);
fprintf(fid_out,['            "Snm":' num2str(0.0,'%16.15E') '\n']);

coeff_name_base = 'BENNU_';

for nn = 2:DEGREE
    for mm = 0:nn
        if mm == 0
            cnamestr = [coeff_name_base 'J' num2str(nn,'%d')];
        else
            cnamestr = [coeff_name_base 'C'];
            if nn < 10
                cnamestr = [cnamestr '0'];
            end
            cnamestr = [cnamestr num2str(nn,'%d')]; %BENNU_C0202
            if mm < 10
                cnamestr = [cnamestr '0'];
            end
            cnamestr = [cnamestr num2str(mm,'%d')]; %BENNU_C0202
            snamestr = cnamestr;
            snamestr(7) = 'S';
        end
        if particles == 1 && nn <= DEGREEp
            for ii = 1:nump
                if ~isempty(strfind(NAMESp{ii},cnamestr))
                    break
                end
                if ii == num
                    error(['Coefficient ' cnamestr 'not found'])
                end
            end
            Cval = VALSp(ii);
            if mm > 0
                for ii = 1:nump
                    if ~isempty(strfind(NAMESp{ii},snamestr))
                        break
                    end
                    if ii == num
                        error(['Coefficient ' cnamestr 'not found'])
                    end
                end
                Sval = VALSp(ii);
            else
                Sval = 0.0;
                Cval = -Cval; % because here they give J values, not Cnn0
            end
        else
            for ii = 1:num
                if ~isempty(strfind(NAMES{ii},cnamestr))
                    break
                end
                if ii == num
                    error(['Coefficient ' cnamestr 'not found'])
                end
            end
            Cval = VALS(ii);
            if mm > 0
                for ii = 1:num
                    if ~isempty(strfind(NAMES{ii},snamestr))
                        break
                    end
                    if ii == num
                        error(['Coefficient ' cnamestr 'not found'])
                    end
                end
                Sval = VALS(ii);
            else
                Sval = 0.0;
                Cval = -Cval; % because here they give J values, not Cnn0
            end
        end

        fprintf(fid_out,'        },\n');
        fprintf(fid_out,'        {\n');
    	fprintf(fid_out,['            "degree":' num2str(nn,'%d') ',\n']);
    	fprintf(fid_out,['            "order":' num2str(mm,'%d') ',\n']);
    	fprintf(fid_out,['            "Cnm":' num2str(Cval,'%16.15E') ',\n']);
    	fprintf(fid_out,['            "Snm":' num2str(Sval,'%16.15E') '\n']);
    end
end
fprintf(fid_out,'        }\n');
fprintf(fid_out,'    ],\n');
fprintf(fid_out,'    "source":[\n');
fprintf(fid_out,'        {\n');
fprintf(fid_out,['            "url":"' sourceURL '",\n']);
fprintf(fid_out,['            "desciption":"' sourceDescription '"\n']);
fprintf(fid_out,'        }\n');
fprintf(fid_out,'    ]\n');
fprintf(fid_out,'}');

fclose(fid_out);

end