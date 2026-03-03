const STAR_PATH =
  "M12 2l2.9 6.6L22 9.5l-5 4.8 1.2 7.2L12 18l-6.2 3.5L7 14.3l-5-4.8 7.1-.9z";

interface StarRatingProps {
  rating: number;
  size?: number;
}

export default function StarRating({ rating, size = 14 }: StarRatingProps) {
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <svg
          key={n}
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill={n <= rating ? "var(--thumbnail-font-color)" : "none"}
          stroke="var(--thumbnail-font-color)"
          strokeWidth={2}
          strokeLinejoin="round"
          style={{ cursor: "pointer" }}
        >
          <path d={STAR_PATH} />
        </svg>
      ))}
    </div>
  );
}
