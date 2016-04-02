load('traces.mat')
for t = [1:length(traces(:))]
  filename=sprintf('output/%05d.png',t)
  plot(traces(1, t, 1), traces(1, t, 2), traces(2, t, 1), traces(2, t, 2), traces(3, t, 1), traces(3, t, 2));axis([-40,40, -20,20])
  print(filename, '-dpng')
endfor
