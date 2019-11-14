#!/usr/bin/perl

$n = 0;
opendir(DIR,".");
@D = readdir(DIR);
closedir(DIR);
foreach $f (sort @D) {
    next unless (-d $f);
    next unless ($f =~ /^r_\d+/);

    open (LOG,"$f/run.log");
    $n++;
    while (<LOG>) {
        next unless /^Time /;
        ($l = $_) =~ s/:/ /g;
        @X = split(" ",$l);
        $t = $X[$#X-1];
#        print "$t\n";
        if (/GPU ChainA/) {
            push ( @{ $time{"ga"} }, $t );
            push(@ga,$X[$#X-1]);
        } elsif (/GPU ChainB/) {
            push ( @{ $time{"gb"} }, $t );
            push(@gb,$X[$#X-1]);
        } elsif (/host Chain0/) {
            push ( @{ $time{"hc"} }, $t );
            push(@hc,$X[$#X-1]);
        } elsif (/simul/) {
            push ( @{ $time{"sm"} }, $t );
            push(@sm,$X[$#X-1]);
        } elsif (/Chain 0/) {
            push ( @{ $time{"c0"} }, $t );
            push(@c0,$X[$#X-1]);
        } elsif (/eventloop/) {
            push ( @{ $time{"el"} }, $t );
            push(@el,$X[$#X-1]);
        }
            


    }
    close(LOG);
}
open (IN,"time.log") || die "can't find time.log\n";
while(<IN>) {
    chop;
    push (@{ $time{"tt"} }, $_);
}
close(IN);
        

foreach $k (sort keys %time ) {
    @D = @{ $time{$k} };
    if ($#D != $n-1) {
        print "error: wrong number of entries in array $k. found $#D should be $n\n";
        exit(1);
    }
    $tot = 0.;
    foreach $i ( @D ) {
        $tot += $i;
    }
    $avg = $tot / $n;
    $avg{$k} = $avg;
#    print "$k $avg\n";
}

# print "$n";
foreach $a (sort keys %avg) {
    printf(",%.3f",$avg{$a});
}
print "\n";
    
