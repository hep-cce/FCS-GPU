#!/usr/bin/perl

if ( $#ARGV == -1 ) {
    $all = 1;
    $proc=".*";
} else {
    $proc = $ARGV[0];
}

#print "proc: $proc\n";

$p = 0;
@X = `cat /proc/cpuinfo`;
foreach (@X) {
    chop;
    if (/^processor\s+:\s*($proc)$/) {
        $pp = $1;
        $p = 1;
    }
    next unless ($p == 1);

    $speed = 0;
#    print "      $pp $_\n";
    next unless /^cpu MHz\s+:\s+(.*)$/;
    if ($all == 1) {
        $p = 0;
        print "$pp $1\n";
    } else {
        $speed = $1;
        print "$proc $speed\n";
        exit 0;
    }
}

