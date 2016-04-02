graphics_toolkit gnuplot
figure('visible','off');

load('tracks.mat')
for t = [1:size(pos)(1)]
  filename=sprintf('output/%05d.png',t);
  axis([-60,60, -35,35])
  plot(pos(t, 1, 1), pos(t, 1, 2), 'rx');
  hold on
  for p = [2:23]
    axis([-60,60, -35,35])
    plot(pos(t, p, 1), pos(t, p, 2), 'b.');
  endfor
  print(filename, '-dpng')
  hold off
endfor  

