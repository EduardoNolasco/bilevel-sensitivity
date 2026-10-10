-- Example: number of runs and mean iterations per status
SELECT status, COUNT(*) AS runs, ROUND(AVG(iterations), 1) AS mean_iterations
FROM runs
GROUP BY status;

-- List all runs of Bard_1991_ex1 with their start, status and iterations --
SELECT x0, status, iterations
FROM runs
WHERE problem='Bard_1991_ex1'
ORDER BY iterations DESC;

-- For each problem: number of runs and total time --
SELECT problem, COUNT(problem), ROUND(SUM(time_s),1)
FROM runs
GROUP BY problem;

-- For every run show the problem, start, f_found, f_known --
-- and the absolute error ABS(f_found - f_known) --
SELECT problem, x0, ROUND(f_found,1) AS f_found, f_known, ROUND(ABS(f_found - f_known),1) AS abs_error
FROM runs
JOIN problems ON problems.name=runs.problem;

-- For each problem, runs and how many reached the known value (error < 0.1) --
SELECT p.name,
       COUNT(*) AS runs,
       SUM(CASE WHEN ABS(r.f_found - p.f_known)<0.1 THEN 1 ELSE 0 END) AS reached
FROM runs AS r
JOIN problems AS p ON p.name=r.problem
GROUP BY p.name;

-- Which problems have no run that reached the known value? --
SELECT p.name,
       COUNT(*) AS runs,
       SUM(CASE WHEN ABS(r.f_found - p.f_known)<0.1 THEN 1 ELSE 0 END) AS reached
FROM runs AS r
JOIN problems AS p ON p.name=r.problem
GROUP BY p.name
HAVING reached = 0;

WITH hits AS (
        SELECT p.name,
       COUNT(*) AS runs,
       SUM(CASE WHEN ABS(r.f_found - p.f_known)<0.1 THEN 1 ELSE 0 END) AS reached
FROM runs AS r
JOIN problems AS p ON p.name=r.problem
GROUP BY p.name
HAVING reached = 0
)
SELECT name, runs
FROM hits
WHERE reached=0;
