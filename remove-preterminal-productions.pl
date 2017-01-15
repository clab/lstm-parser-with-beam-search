#!/usr/bin/perl -w
use strict;
my $x = <>;
while($x) {
  my $y = <>;
  unless ($y =~ /SHIFT/) {
    print $x;
  }
  $x = $y;
}
print $x;

