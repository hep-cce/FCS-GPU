#!/usr/bin/perl

use Digest::MD5 qw(md5 md5_hex md5_base64);

#
## Extracts information about system, including details of CPU and 
##   GPU hardware and driver versions
## Will produce a json output
#
# author: C. Leggett
# date:   2022/06/01
#

$HOST = `hostname -A`;
$HOST =~ s/\s*\n$//;

$MSG = "";

#
## Use lsb_release to extract OS information
#
$REL = "unknown";
$LSBREL = `which lsb_release 2> /dev/null`;
if ( $? == 0 ) {
    chop $LSBREL;
    $rel = `$LSBREL -d`;
    chop $rel;
    ($REL = $rel) =~ s/Description:\s*//;
    $REL =~ s/\s*$//;
};

#
## Use lscpu to extract CPU information
#
$LSCPU = `which lscpu`;
chop $LSCPU;
if ( $? != 0 ) {
    print STDERR "ERROR: no lscpu found\n";
    exit(1);
}

@CPU = `$LSCPU`;

# print "@CPU\n";
foreach (@CPU) {
    if (/Model name:\s+(.*)/) {
        $CPU{"cpu_id"} = $1;
    } elsif (/^CPU\(s\):\s+(.*)/) {
        $CPU{"cores"} = $1;
    } elsif (/^Socket\(s\):\s+(.*)/) {
        $CPU{"sockets"} = $1;
    } elsif (/^Thread\(s\) per core:\s+(.*)/) {
        $CPU{"threads_per_core"} = $1;
    } elsif (/^CPU max MHz:\s+(.*)/) {
        $CPU{"max_clock"} = $1;
    }
}



$MSG .= "{\n  \"hostname\": \"$HOST\",\n";
$MSG .= "  \"OS\": \"$REL\",\n";
foreach $k (keys %CPU) {
    $MSG .= "  \"$k\": \"$CPU{$k}\",\n";
}
$MSG .= "  \"GPUs\": [\n";

#
## Use nvidia-smi to extract details of NVIDIA GPUs
#
$NVS= `which nvidia-smi 2> /dev/null`;
if ( $? == 0 ) {
    @NV = `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.current.graphics,clocks.current.memory,clocks.current.sm --format=csv,noheader`;
    if ( $? != 0 ) {
        @NV = `nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.current.graphics,clocks.current.memory,clocks.current.sm --format=csv,noheader`;
        foreach (@NV) {
            chop $_;
            ($GPU{"gpu_id"},$GPU{"driver"},$GPU{"memory"},$GPU{"memory_clock"},$GPU{"graphics_clock"},$GPU{"sm_clock"}) = split(",",$_);
            
            $MSG .= "    {\n";
            foreach $k (sort keys %GPU) {
                $v = $GPU{$k};
                $v =~ s/^\s*//;
                $MSG .= "        \"$k\": \"$v\",\n";
            }
            $MSG .= "    },\n";
        }
    } else {            
        foreach (@NV) {
            chop $_;
            ($GPU{"gpu_id"},$GPU{"driver"},$GPU{"memory"},$GPU{"compute_capability"},$GPU{"memory_clock"},$GPU{"graphics_clock"},$GPU{"sm_clock"}) = split(",",$_);
            
            $MSG .= "    {\n";
            foreach $k (sort keys %GPU) {
                $v = $GPU{$k};
                $v =~ s/^\s*//;
                $MSG .= "        \"$k\": \"$v\",\n";
            }
            $MSG .= "    },\n";
        }
    }
} else {
#    print STDERR "no nvidia-smi found\n";
}

#
## Use rocm-smi to extract details of AMD GPUs
#
$ROC = `which rocm-smi 2> /dev/null`;
if ( $? == 0 ) {
    chop $ROC;
    @X = `$ROC | grep -v "========" | grep -v "GPU"`;
    $N = $#X - 1;
    foreach $i ( 1 .. $N ) {
        $MSG .= "    {\n";
        $n = `rocm-smi -d $i --showproductname | grep series`;
        $n =~ /Card series:\s+(.*)$/;
        $name = $1;
        $d = `rocm-smi -d $i --showdriverversion | grep version`;
        $d =~ /version:\s+(.*)$/;
        $dv = $1;
        $MSG .= "      \"gpu_id\": \"$name\",\n";
        $MSG .= "      \"driver\": \"$dv\",\n";

        $fc = 0;
        $mc = 0;
        $sc = 0;
        $so = 0;
        
        @Y = `$ROC -d $i -c`;
        foreach ( @Y ) {
            if (/fclk.*\((.*)\)/) {
                $fc = $1;
            }
            if (/mclk.*\((.*)\)/) {
                $mc = $1;
            }
            if (/sclk.*\((.*)\)/) {
                $sc = $1;
            }
            if (/socclk.*\((.*)\)/) {
                $so = $1;
            }
        }

        $MSG .= "      \"fclk\": \"$fc\",\n";
        $MSG .= "      \"mclk\": \"$mc\",\n";
        $MSG .= "      \"sclk\": \"$sc\",\n";
        $MSG .= "      \"socclk\": \"$so\",\n";

        $MSG .= "    },\n";
    }
}

$MSG .= "  ],\n";
$MSG .= "}\n";

print $MSG;

$digest = md5_hex($MSG);
$short = substr($digest, 0, 8);

print "md5 sum: $digest $short\n";
