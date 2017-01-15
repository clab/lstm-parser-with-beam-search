#!/usr/bin/perl -w
use strict;
die "Usage: $0 dict.txt < oracle.txt\n" unless scalar @ARGV == 1;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

my %dict;
print STDERR "Reading dictionary from $ARGV[0]\n";
open F, "<$ARGV[0]" or die "Can't read $ARGV[0]: $!";
binmode(F,":utf8");
while(<F>) {
  chomp;
  $dict{$_} = 1;
}
close F;

my $sc = 0;
my %ud = ();
while(<STDIN>) {
  chomp;
  # line format should be
  # INDEX_OF_WORD word
  # INDEX_OF_WORD otherword
  my ($wordidx, $word, $x) = split /\s+/;
  die unless defined $wordidx;
  die unless defined $word;
  die if defined $x;
  $x = unkify($_, $wordidx);
  print "$x\n";
}

# algorithm taken from edu/berkeley/nlp/PCFGLA/SophisticatedLexicon.java
# $word is the input word
# $loc is its index in the input sentence (0 or >0)
sub unkify {
  my ($word, $loc) = @_;
  my $sb = 'UNK';
  my $wlen = length($word);
  my $numCaps = 0;
  my $hasDigit = 0;
  my $hasDash = 0;
  my $hasLower = 0;
  for (my $i = 0; $i < $wlen; $i++) {
    my $ch = substr($word, $i, 1);
    $hasDigit = ($ch =~ /\d/);
    $hasDash = ($ch =~ /-/);
    if ($ch =~ /\p{L}/) {
      $hasLower = ($ch eq lc $ch);
      $numCaps++ if ($ch eq uc $ch);
    }
  }
  my $ch0 = substr($word, 0, 1);
  my $lowered = lc $word;
  if ($ch0 eq uc $ch0 && $ch0 =~ /\p{L}/) {
    if ($loc == 0 && $numCaps == 1) {
      $sb .= '-INITC';
      if ($dict{$lowered}) { $sb .= '-KNOWNLC'; }
    } else {
      $sb .= '-CAPS';
    }
  } elsif (!($ch0 =~ /\p{L}/) && $numCaps > 0) {
    $sb .= '-CAPS';
  } elsif ($hasLower) {
    $sb .= '-LC';
  }
  $sb .= '-NUM' if $hasDigit;
  $sb .= '-DASH' if $hasDash;
  if ($lowered =~ /s$/ && $wlen >= 3) {
    my $ch2 = substr($lowered, $wlen - 2, 1);
    unless ($ch2 eq 's' || $ch2 eq 'i' || $ch2 eq 'u') {
      $sb .= '-s';
    }
  } elsif ($wlen >= 5 && !$hasDash && !($hasDigit && $numCaps > 0)) {
    $sb .= '-ed' if $lowered =~ /ed$/;
    $sb .= '-ing' if $lowered =~ /ing$/;
    $sb .= '-ion' if $lowered =~ /ion$/;
    $sb .= '-er' if $lowered =~ /er$/;
    $sb .= '-est' if $lowered =~ /est$/;
    $sb .= '-ly' if $lowered =~ /ly$/;
    $sb .= '-ity' if $lowered =~ /ity$/;
    $sb .= '-al' if $lowered =~ /al$/;
    unless ($sb =~ /y$/) { $sb .= '-y' if $lowered =~ /y$/; }
  }
  return $sb;
}
